"""Tests for the deterministic cleaning layer.

These matter more than usual for this project: the whole architectural claim
is that transformation is reproducible because it is code rather than a
prompt. That claim is only worth making if it is tested.
"""

from datetime import date

import pytest

import normalize as N


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-02-26", date(2026, 2, 26)),
        ("2026-2-6", date(2026, 2, 6)),
        ("26/02/2026", date(2026, 2, 26)),   # day > 12 forces day-first
        ("02/26/2026", date(2026, 2, 26)),   # second part > 12 forces month-first
        ("15.03.2026", date(2026, 3, 15)),
        ("2026-02-26T09:30:00Z", date(2026, 2, 26)),
        ("45930", date(2025, 9, 30)),        # Excel serial
        ("Sep 2026", date(2026, 9, 1)),
        ("15 Jan 2026", date(2026, 1, 15)),
        ("Jan 15, 2026", date(2026, 1, 15)),
        ("Q4 2026", date(2026, 10, 1)),
        ("2026 Q2", date(2026, 4, 1)),
        ('{"date":"2026-03-15"}', date(2026, 3, 15)),
        (date(2026, 5, 1), date(2026, 5, 1)),
    ],
)
def test_parse_date_formats(raw, expected):
    assert N.parse_date(raw).value == expected


def test_ambiguous_date_is_flagged_not_silently_guessed():
    """03/04 could be 3 Apr or 4 Mar. We pick one AND say we did."""
    result = N.parse_date("03/04/2026")
    assert result.value == date(2026, 4, 3)
    assert result.issue == N.DATE_AMBIGUOUS


def test_missing_and_malformed_dates_are_distinguished():
    assert N.parse_date("").issue == N.DATE_MISSING
    assert N.parse_date("n/a").issue == N.DATE_MISSING
    assert N.parse_date("TBD").issue == N.DATE_MISSING
    assert N.parse_date("sometime soon").issue == N.DATE_UNPARSEABLE


def test_nat_and_nan_read_as_missing_not_unparseable():
    pd = pytest.importorskip("pandas")
    assert N.parse_date(pd.NaT).issue == N.DATE_MISSING
    assert N.parse_date(float("nan")).issue == N.DATE_MISSING
    assert N.parse_money(pd.NaT).value is None


def test_implausible_serial_rejected():
    assert N.parse_date("99999").issue == N.DATE_IMPLAUSIBLE


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("489360", 489360.0),
        ("489,360", 489360.0),
        ("₹1,01,060", 101060.0),          # Indian digit grouping
        ("$1.2M", 1_200_000.0),
        ("450k", 450_000.0),
        ("INR 2,500", 2500.0),
        (751473450, 751473450.0),
    ],
)
def test_parse_money(raw, expected):
    assert N.parse_money(raw).value == pytest.approx(expected)


def test_accounting_negative_is_parsed_and_flagged():
    result = N.parse_money("(82,907.30)")
    assert result.value == pytest.approx(-82907.30)
    assert result.issue == N.MONEY_NEGATIVE


def test_missing_money_is_none_not_zero():
    """Zero and 'not recorded' are different facts and must not be conflated."""
    assert N.parse_money("").value is None
    assert N.parse_money("n/a").value is None
    assert N.parse_money("0").value == 0.0


# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------


def test_sector_canonicalisation_handles_case_and_noise():
    assert N.parse_sector("mining").value == "Mining"
    assert N.parse_sector("MINING").value == "Mining"
    assert N.parse_sector("  Mining  ").value == "Mining"
    assert N.parse_sector("renewables sector").value == "Renewables"
    assert N.parse_sector("coal").value == "Mining"


def test_unknown_sector_is_reported_not_guessed():
    """A sector we do not recognise must not be forced into the nearest bucket."""
    result = N.parse_sector("Healthcare")
    assert result.value is None
    assert result.issue == N.CATEGORY_UNMAPPED


def test_energy_expands_to_the_sectors_that_actually_exist():
    sectors, note = N.resolve_sector_query("energy")
    assert sectors == ["Renewables", "Powerline"]
    assert "not a sector in this data" in note


def test_sector_group_survives_noise_words():
    sectors, _ = N.resolve_sector_query("Energy Sector")
    assert sectors == ["Renewables", "Powerline"]


def test_unmatchable_sector_returns_empty_with_explanation():
    sectors, note = N.resolve_sector_query("Hospitality")
    assert sectors == []
    assert "does not match any sector" in note


def test_deal_stage_letter_prefixes_are_preserved():
    assert N.parse_deal_stage("F. Negotiations").value == "F. Negotiations"
    assert N.parse_deal_stage("negotiations").value == "F. Negotiations"
    assert N.parse_deal_stage("won").value == "G. Project Won"


def test_billing_status_typo_folds_into_one_bucket():
    """The source spells this 'BIlled'; it must not become its own category."""
    assert N.parse_billing_status("BIlled").value == "Billed"
    assert N.parse_billing_status("Billed").value == "Billed"


def test_per_visit_billing_reads_as_partial_not_full():
    assert N.parse_invoice_status("Billed- Visit 3").value == "Partially Billed"
    assert N.parse_invoice_status("Billed- Visit 7").value == "Partially Billed"
    assert N.parse_invoice_status("Fully Billed").value == "Fully Billed"


def test_funnel_groups_are_disjoint_and_complete():
    """Every canonical stage belongs to exactly one funnel group."""
    grouped = (
        N.OPEN_DEAL_STAGES + N.WON_DEAL_STAGES
        + N.LOST_DEAL_STAGES + N.HOLD_DEAL_STAGES
    )
    assert sorted(grouped) == sorted(N.CANONICAL_DEAL_STAGES)
    assert len(grouped) == len(set(grouped)), "a stage appears in two groups"


def test_won_stages_are_not_counted_as_open_pipeline():
    """The easiest way to double-count revenue on this board."""
    for stage in N.WON_DEAL_STAGES:
        assert stage not in N.OPEN_DEAL_STAGES


# --------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------


def test_quantity_splits_amount_from_unit():
    q = N.parse_quantity("5360 HA")
    assert q.value.amount == 5360.0
    assert q.value.unit == "HA"


def test_bare_quantity_is_usable_but_flagged_unitless():
    q = N.parse_quantity("4")
    assert q.value.amount == 4.0
    assert q.value.unit is None
    assert q.issue == N.QTY_NO_UNIT


def test_acre_spellings_collapse_to_one_unit():
    """ACR / ACRE / ACERS are the same unit spelled three ways in the source."""
    assert N.parse_quantity("10 ACR").value.unit == "ACRE"
    assert N.parse_quantity("10 acres").value.unit == "ACRE"
    assert N.parse_quantity("10 ACERS").value.unit == "ACRE"


def test_route_km_stays_distinct_from_km():
    assert N.parse_quantity("12 RKM").value.unit == "RKM"
    assert N.parse_quantity("12 km").value.unit == "KM"


# --------------------------------------------------------------------------
# Caveats
# --------------------------------------------------------------------------


def test_caveats_group_and_count_rather_than_listing_every_row():
    cav = N.CaveatCollector(noun="deal")
    for name in ["A", "B", "C", "D", "E"]:
        cav.record("deal value", N.MONEY_MISSING, name)
    texts = cav.to_text()
    assert len(texts) == 1
    assert "5 deals" in texts[0]


def test_caveat_examples_are_deduplicated():
    """Names repeat heavily; 'e.g. Sakura, Sakura, Sakura' is useless."""
    cav = N.CaveatCollector(noun="deal")
    for _ in range(10):
        cav.record("deal value", N.MONEY_MISSING, "Sakura")
    assert cav.caveats[0].examples == ["Sakura"]


def test_hard_gaps_sort_before_soft_notes():
    cav = N.CaveatCollector(noun="deal")
    cav.record("sector", N.CATEGORY_FUZZY, "A")          # soft
    cav.record("deal value", N.MONEY_MISSING, "B")       # hard
    assert cav.caveats[0].issue == N.MONEY_MISSING


def test_clean_parse_records_no_caveat():
    cav = N.CaveatCollector()
    cav.record("sector", None, "A")
    assert cav.caveats == []


# --------------------------------------------------------------------------
# Period resolution
# --------------------------------------------------------------------------


def test_quarter_bounds():
    import analytics as A

    start, end, described = A.resolve_period("this quarter", today=date(2026, 8, 30))
    assert (start, end) == (date(2026, 7, 1), date(2026, 9, 30))
    assert "Q3 2026" in described


def test_last_quarter_wraps_across_new_year():
    import analytics as A

    start, end, _ = A.resolve_period("last quarter", today=date(2026, 1, 15))
    assert (start, end) == (date(2025, 10, 1), date(2025, 12, 31))


def test_unrecognised_period_is_reported_not_silently_ignored():
    import analytics as A

    _, _, described = A.resolve_period("whenever", today=date(2026, 8, 30))
    assert "could not interpret" in described
