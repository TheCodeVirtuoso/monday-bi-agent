"""Tests over the real source data, end to end through normalisation.

These assert on facts about the actual trackers in ``data/``, so they double
as a regression check on the source: if someone re-exports the boards and the
shape changes, these fail loudly instead of the agent quietly reporting
different numbers.
"""

import asyncio

import pytest

import analytics as A
import data_source as DS


@pytest.fixture(scope="module")
def boards():
    return asyncio.run(DS.load_all())


# --------------------------------------------------------------------------
# Junk-row handling
# --------------------------------------------------------------------------


def test_embedded_header_rows_are_dropped(boards):
    """The deals tracker has two rows that are copies of its own header."""
    deals = boards["deals"]
    assert deals.drop_reasons.get("repeated header row") == 2
    assert all(r["deal_name"] not in ("Deal Name",) for r in deals.records)


def test_nameless_rows_are_dropped(boards):
    assert boards["deals"].drop_reasons.get("no deal name") == 2


def test_no_junk_category_survives_into_the_records(boards):
    """A header row leaking through would appear as its own sector/stage."""
    sectors = {r["sector"] for r in boards["deals"].records}
    stages = {r["stage"] for r in boards["deals"].records}
    assert "Sector/service" not in sectors
    assert "Deal Stage" not in stages


def test_boards_are_cached_process_wide_not_per_caller():
    """One copy per process, or memory and monday's rate limit both suffer.

    Each chat session used to hold its own board copy and every health check
    forced a fresh 522-item fetch, which killed a small instance.
    """
    DS.clear_cache()
    first = asyncio.run(DS.get_cached_boards())
    second = asyncio.run(DS.get_cached_boards())
    assert first is second, "get_cached_boards returned a fresh load"
    assert first["deals"] is second["deals"]


def test_cache_can_be_forced_to_refresh():
    DS.clear_cache()
    first = asyncio.run(DS.get_cached_boards())
    refreshed = asyncio.run(DS.get_cached_boards(refresh=True))
    assert refreshed is not first
    assert len(refreshed["deals"].records) == len(first["deals"].records)


def test_row_accounting_adds_up(boards):
    for board in boards.values():
        assert len(board.records) + board.rows_dropped == board.rows_in_source


# --------------------------------------------------------------------------
# Funnel semantics
# --------------------------------------------------------------------------


def test_every_deal_lands_in_a_known_funnel_group(boards):
    groups = {r["stage_group"] for r in boards["deals"].records}
    assert groups <= {"open", "won", "lost", "hold", "unknown"}


def test_open_pipeline_excludes_won_deals(boards):
    snapshot = A.pipeline_snapshot(boards["deals"].records)
    stages = {row["stage"] for row in snapshot["by_stage"]}
    assert not stages & set(
        ["G. Project Won", "H. Work Order Received", "Project Completed"]
    )


def test_win_rate_denominator_excludes_open_deals(boards):
    funnel = A.funnel_snapshot(boards["deals"].records)
    counts = funnel["counts"]
    assert funnel["decided"] == counts.get("won", 0) + counts.get("lost", 0)
    assert "open" in counts and counts["open"] > 0


# --------------------------------------------------------------------------
# Coverage and honesty of aggregates
# --------------------------------------------------------------------------


def test_money_stats_report_coverage_not_just_a_sum(boards):
    stats = A.money_stats(boards["deals"].records, "value_inr")
    assert stats["count_with_value"] < stats["count_records"]
    assert 0 < stats["coverage_pct"] < 100
    assert any("floor, not a total" in w for w in stats["warnings"])


def test_concentration_is_flagged_on_skewed_values(boards):
    stats = A.money_stats(boards["deals"].records, "value_inr")
    assert stats["top_2_share_pct"] >= 25
    assert any("median" in w for w in stats["warnings"])


@pytest.mark.parametrize(
    "value,expected",
    [
        (922_173_105.65, "₹92.22 Cr"),   # the first live run said "₹9.2 Cr"
        (36_291_748.87, "₹3.63 Cr"),
        (1_101_060, "₹11.01 L"),
        (100_000, "₹1.00 L"),
        (99_999, "₹99,999"),
        (10_000_000, "₹1.00 Cr"),
        (-82_907.30, "-₹82,907"),
        (0, "₹0"),
        (None, None),
    ],
)
def test_inr_formatting_is_done_in_code_not_by_the_model(value, expected):
    """Rupee->crore conversion is an order-of-magnitude trap for an LLM."""
    assert A.format_inr(value) == expected


def test_money_stats_ship_preformatted_display_strings(boards):
    """The agents quote these verbatim instead of converting."""
    stats = A.money_stats(boards["deals"].records, "value_inr")
    for key in ("sum", "median", "mean", "min", "max"):
        assert stats[f"{key}_display"], key
        assert stats[f"{key}_display"].lstrip("-").startswith("₹")


def test_breakdown_rows_carry_display_strings(boards):
    rows = A.breakdown(boards["deals"].records, "sector", "value_inr")
    assert any(r["sum_display"] for r in rows)


def test_receivables_snapshot_ships_display_strings(boards):
    snap = A.receivables_snapshot(boards["work_orders"].records)
    assert snap["total_receivable_incl_gst_display"].startswith("₹")
    assert all("receivable_display" in r for r in snap["top_receivables"])


# --------------------------------------------------------------------------
# Data recency
# --------------------------------------------------------------------------


def test_stale_board_is_flagged_so_zero_is_not_read_as_no_business(boards):
    """Expected close dates end Apr 2026; 'this quarter' correctly finds none."""
    texts = " ".join(c["text"] for c in boards["deals"].caveats)
    assert "expected close date values run" in texts


def test_out_of_range_period_explains_itself(boards):
    from datetime import date as _date

    kept, notes = A.filter_records(
        boards["deals"].records,
        date_field="expected_close_date",
        date_from=_date(2030, 1, 1),
        date_to=_date(2030, 12, 31),
    )
    assert kept == []
    assert any("outside that range" in n for n in notes)
    assert any("not an absence of" in n for n in notes)


def test_date_range_helper_reports_coverage(boards):
    rng = A.date_range(boards["deals"].records, "expected_close_date")
    assert rng["earliest"] < rng["latest"]
    assert rng["count_with_date"] < rng["count_records"]


def test_empty_selection_reports_no_value_rather_than_zero(boards):
    """Zero records must not silently produce a sum of 0."""
    stats = A.money_stats([], "value_inr")
    assert stats["sum"] is None
    assert stats["count_with_value"] == 0


def test_board_carries_structural_caveats(boards):
    texts = " ".join(c["text"] for c in boards["deals"].caveats)
    assert "no recorded value" in texts
    assert "concentrated" in texts


# --------------------------------------------------------------------------
# Date filtering
# --------------------------------------------------------------------------


def test_date_filter_discloses_what_it_excluded(boards):
    """Undated records dropped by a range filter must be reported."""
    from datetime import date

    _, notes = A.filter_records(
        boards["deals"].records,
        date_field="expected_close_date",
        date_from=date(2026, 1, 1),
        date_to=date(2026, 12, 31),
    )
    assert any("no expected close date" in n for n in notes)


# --------------------------------------------------------------------------
# The cross-board constraint
# --------------------------------------------------------------------------


def test_client_codes_genuinely_do_not_overlap(boards):
    """The premise behind refusing a client-level join."""
    deal_clients = {r["client_code"] for r in boards["deals"].records if r["client_code"]}
    wo_clients = {r["customer_code"] for r in boards["work_orders"].records if r["customer_code"]}
    assert deal_clients and wo_clients
    assert not (deal_clients & wo_clients)


def test_deal_names_are_not_unique_so_cannot_be_a_join_key(boards):
    names = [r["deal_name"] for r in boards["deals"].records]
    assert len(set(names)) < len(names)


@pytest.mark.parametrize("bad_key", ["client_code", "deal_name"])
def test_unsafe_joins_are_refused_with_a_reason(boards, bad_key):
    with pytest.raises(A.UnsafeJoinError) as exc:
        A.cross_board_view(
            boards["deals"].records, boards["work_orders"].records, by=bad_key
        )
    assert bad_key in str(exc.value)


def test_cross_board_ranks_in_code_rather_than_by_string_comparison(boards):
    """'₹69.77 L' vs '₹9.35 Cr' is easy to get backwards; rank it here."""
    view = A.cross_board_view(
        boards["deals"].records, boards["work_orders"].records, by="sector"
    )
    biggest = max(
        (r for r in view["rows"] if r["work_order_value_sum"]),
        key=lambda r: r["work_order_value_sum"],
    )
    assert view["largest_by_work_order_value"] == biggest["sector"]
    assert view["rows"][0]["deal_value_sum_display"].startswith("₹")


def test_cross_board_states_it_covers_all_stages_not_just_open(boards):
    view = A.cross_board_view(
        boards["deals"].records, boards["work_orders"].records, by="sector"
    )
    assert "NOT open-pipeline" in view["population"]


def test_owner_join_works_and_flags_one_sided_owners(boards):
    view = A.cross_board_view(
        boards["deals"].records, boards["work_orders"].records, by="owner"
    )
    assert view["aligned_on"] == "owner"
    assert view["rows"]
    assert any("share no client or deal identifier" in c for c in view["caveats"])


def test_sector_join_reports_deals_only_sectors(boards):
    """Five sectors exist on the deals board and never on work orders."""
    view = A.cross_board_view(
        boards["deals"].records, boards["work_orders"].records, by="sector"
    )
    deals_only = [r["sector"] for r in view["rows"] if not r["work_order_count"]]
    assert "Aviation" in deals_only or "DSP" in deals_only


# --------------------------------------------------------------------------
# Receivables
# --------------------------------------------------------------------------


def test_receivables_snapshot_flags_negative_to_bill(boards):
    snap = A.receivables_snapshot(boards["work_orders"].records)
    assert snap["negative_to_bill_count"] > 0
    assert "over-billing" in snap["negative_to_bill_note"]


def test_receivables_snapshot_states_its_gst_basis(boards):
    snap = A.receivables_snapshot(boards["work_orders"].records)
    assert "INCLUSIVE" in snap["basis_note"] and "EXCLUSIVE" in snap["basis_note"]


def test_ar_priority_accounts_are_identified(boards):
    snap = A.receivables_snapshot(boards["work_orders"].records)
    assert snap["ar_priority_count"] == 10


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------


def test_deals_tool_applies_period_and_says_so(boards):
    import tools as T

    out = T.run_deals_tool(
        "deals_query",
        {"stage_groups": ["open"], "period": "this quarter"},
        boards["deals"],
    )
    assert any("period read as" in n for n in out["notes"])
    assert "value" in out and "by_stage" in out


def test_work_orders_receivables_tool_returns_board_caveats(boards):
    import tools as T

    out = T.run_work_order_tool(
        "work_orders_snapshot", {"kind": "receivables"}, boards["work_orders"]
    )
    assert out["total_receivable_incl_gst"] > 0
    assert out["board_caveats"]
