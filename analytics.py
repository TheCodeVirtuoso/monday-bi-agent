"""Deterministic aggregation over normalized records.

No LLM touches arithmetic. Every number a founder sees is produced here, by
ordinary Python, and is reproducible.

The governing idea is that **an aggregate must carry its own provenance**.
``sum`` on its own is a lie when 61% of the rows have no value; every
statistic below therefore reports how many records actually contributed
(``coverage``) alongside the figure itself, and the agents are instructed
never to quote one without the other.
"""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from typing import Any, Sequence

import normalize as N

Record = dict[str, Any]

# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------


def _as_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value:
        parsed = N.parse_date(value)
        return parsed.value if isinstance(parsed.value, date) else None
    return None


def filter_records(
    records: Sequence[Record],
    *,
    sectors: Sequence[str] | None = None,
    owners: Sequence[str] | None = None,
    date_field: str | None = None,
    date_from: object = None,
    date_to: object = None,
    match: dict[str, Sequence[str]] | None = None,
) -> tuple[list[Record], list[str]]:
    """Filter records, returning ``(kept, notes)``.

    ``notes`` explains anything the caller must disclose — most importantly
    how many records a date filter *excluded because they had no date*. That
    exclusion is the single easiest way to silently understate a pipeline, so
    it is always reported rather than left implicit.
    """
    kept = list(records)
    notes: list[str] = []

    if sectors:
        wanted = {s.lower() for s in sectors}
        kept = [r for r in kept if (r.get("sector") or "").lower() in wanted]

    if owners:
        wanted = {o.lower() for o in owners}
        kept = [r for r in kept if (r.get("owner") or "").lower() in wanted]

    for field_name, allowed in (match or {}).items():
        allowed_set = {str(a).lower() for a in allowed}
        kept = [r for r in kept if str(r.get(field_name) or "").lower() in allowed_set]

    if date_field and (date_from or date_to):
        lo, hi = _as_date(date_from), _as_date(date_to)
        undated = [r for r in kept if not r.get(date_field)]
        dated = []
        for r in kept:
            d = _as_date(r.get(date_field))
            if d is None:
                continue
            if lo and d < lo:
                continue
            if hi and d > hi:
                continue
            dated.append(r)
        if undated:
            notes.append(
                f"{len(undated)} record(s) have no {date_field.replace('_', ' ')} "
                f"and were excluded from this date range — they may still be relevant"
            )

        # An empty result from a date filter is ambiguous: it can mean "nothing
        # is happening" or "you asked about a window this data doesn't cover".
        # Those lead to opposite conclusions, so report the range that DOES
        # exist rather than leaving the reader to assume the former.
        if not dated:
            available = [
                d for d in (_as_date(r.get(date_field)) for r in records) if d
            ]
            if available:
                notes.append(
                    f"NO records fall in this window. The {date_field.replace('_', ' ')} "
                    f"values in this board actually run from {min(available)} to "
                    f"{max(available)} — the requested period is outside that range, "
                    f"so this is a gap in the data's coverage, not an absence of "
                    f"business. Report it that way and suggest a period inside the "
                    f"available range."
                )
            else:
                notes.append(
                    f"no record in this board has a {date_field.replace('_', ' ')} "
                    f"at all, so no date filter can return anything"
                )
        kept = dated

    return kept, notes


def date_range(records: Sequence[Record], field: str) -> dict:
    """Earliest and latest value of a date field, with coverage."""
    values = [d for d in (_as_date(r.get(field)) for r in records) if d]
    return {
        "field": field,
        "count_with_date": len(values),
        "count_records": len(records),
        "earliest": min(values).isoformat() if values else None,
        "latest": max(values).isoformat() if values else None,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def format_inr(value: float | int | None) -> str | None:
    """Render a rupee amount in Indian crore/lakh notation.

    This exists so no model ever performs the conversion. Rupee -> crore is a
    divide by 10,000,000 and an LLM asked to do it inline gets the magnitude
    wrong often enough to matter — the first live run reported ₹92.2 Cr as
    "₹9.2 Cr". Every money figure leaves this module pre-formatted, and the
    agents are told to quote the string rather than convert the number.
    """
    if value is None:
        return None
    negative = value < 0
    v = abs(float(value))

    if v >= 1e7:
        text = f"₹{v / 1e7:,.2f} Cr"
    elif v >= 1e5:
        text = f"₹{v / 1e5:,.2f} L"
    else:
        text = f"₹{v:,.0f}"
    return ("-" + text) if negative else text


def _tagged(value: float | int | None, basis: str) -> str | None:
    """Format an amount with its basis attached to the string itself."""
    text = format_inr(value)
    return f"{text} ({basis})" if text else None


def _with_display(stats: dict, keys: Sequence[str]) -> dict:
    """Attach a pre-formatted ``*_display`` string beside each money key."""
    for key in keys:
        stats[f"{key}_display"] = format_inr(stats.get(key))
    return stats


def money_stats(records: Sequence[Record], field: str) -> dict:
    """Summarise a money column, always alongside how complete it is."""
    total_rows = len(records)
    values = [r[field] for r in records if isinstance(r.get(field), (int, float))]
    n = len(values)

    if not n:
        return {
            "count_records": total_rows,
            "count_with_value": 0,
            "coverage_pct": 0.0,
            "sum": None,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "note": "no records in this selection carry a value",
        }

    ordered = sorted(values, reverse=True)
    total = float(sum(values))
    stats = {
        "count_records": total_rows,
        "count_with_value": n,
        "coverage_pct": round(100 * n / total_rows, 1) if total_rows else 0.0,
        "sum": total,
        "median": float(statistics.median(values)),
        "mean": total / n,
        "min": float(min(values)),
        "max": float(max(values)),
        "top_2_share_pct": round(100 * sum(ordered[:2]) / total, 1) if total else None,
    }

    # Only the figures an answer actually quotes get a display string; min
    # and max keep their raw values for anyone who needs them.
    _with_display(stats, ["sum", "median", "mean"])

    warnings = []
    if stats["coverage_pct"] < 90:
        warnings.append(
            f"only {n} of {total_rows} records ({stats['coverage_pct']}%) carry a "
            f"value — the sum is a floor, not a total"
        )
    if stats["top_2_share_pct"] and stats["top_2_share_pct"] >= 25:
        warnings.append(
            f"the 2 largest records are {stats['top_2_share_pct']}% of the sum; "
            f"the mean ({stats['mean_display']}) is skewed, the median "
            f"({stats['median_display']}) is the honest centre"
        )
    stats["warnings"] = warnings
    return stats


def breakdown(
    records: Sequence[Record],
    by: str,
    value_field: str | None = None,
    top: int | None = None,
    include_median: bool = False,
) -> list[dict]:
    """Group records by a field, with per-group count and value coverage.

    Per-group medians are off by default. They are seldom what a founder
    asks for, and every field here is multiplied by the number of groups and
    the number of breakdowns in a payload the model has to read.
    """
    groups: dict[str, list[Record]] = {}
    for r in records:
        key = r.get(by)
        groups.setdefault("(not recorded)" if key in (None, "") else str(key), []).append(r)

    rows = []
    for key, items in groups.items():
        row: dict[str, Any] = {by: key, "count": len(items)}
        if value_field:
            vals = [
                i[value_field] for i in items
                if isinstance(i.get(value_field), (int, float))
            ]
            row["count_with_value"] = len(vals)
            row["sum"] = float(sum(vals)) if vals else None
            row["sum_display"] = format_inr(row["sum"])
            if include_median:
                row["median"] = float(statistics.median(vals)) if vals else None
                row["median_display"] = format_inr(row["median"])
        rows.append(row)

    rows.sort(key=lambda r: (r.get("sum") is None, -(r.get("sum") or 0), -r["count"]))
    return rows[:top] if top else rows


# --------------------------------------------------------------------------
# Domain snapshots
# --------------------------------------------------------------------------


def pipeline_snapshot(deals: Sequence[Record]) -> dict:
    """Open pipeline only — stages A-F. Won deals are delivery, not pipeline."""
    open_deals = [d for d in deals if d.get("stage_group") == "open"]
    stats = money_stats(open_deals, "value_inr")
    return {
        "definition": (
            "open pipeline = stages A-F (Lead Generated through Negotiations). "
            "Stages G-K and Project Completed are already won and are counted "
            "under delivery, not pipeline."
        ),
        "open_deal_count": len(open_deals),
        "value": stats,
        "by_stage": breakdown(open_deals, "stage", "value_inr", top=8),
        "by_sector": breakdown(open_deals, "sector", "value_inr", top=8),
        "by_owner": breakdown(open_deals, "owner", "value_inr", top=8),
        "late_stage_count": sum(
            1 for d in open_deals if d.get("stage") in N.LATE_STAGE_DEALS
        ),
    }


def funnel_snapshot(deals: Sequence[Record]) -> dict:
    """Counts by funnel group, so win/loss ratios are computable."""
    groups: dict[str, int] = {}
    for d in deals:
        g = d.get("stage_group") or "unknown"
        groups[g] = groups.get(g, 0) + 1

    won, lost = groups.get("won", 0), groups.get("lost", 0)
    decided = won + lost
    return {
        "counts": groups,
        "decided": decided,
        "win_rate_pct": round(100 * won / decided, 1) if decided else None,
        "win_rate_basis": (
            f"{won} won / {decided} decided (won + lost); open and on-hold "
            f"deals are excluded from the denominator"
        ),
    }


def delivery_snapshot(work_orders: Sequence[Record]) -> dict:
    """Execution status of the work-order book."""
    active = [w for w in work_orders if w.get("execution_group") == "active"]
    stalled = [w for w in work_orders if w.get("execution_group") == "stalled"]
    return {
        "total_work_orders": len(work_orders),
        "active_count": len(active),
        "stalled_count": len(stalled),
        "done_count": sum(1 for w in work_orders if w.get("execution_group") == "done"),
        "active_value": money_stats(active, "amount_excl_gst"),
        "stalled_detail": [
            {
                "wo_id": w["wo_id"],
                "deal_name": w.get("deal_name"),
                "sector": w.get("sector"),
                "execution_status": w.get("execution_status"),
                "amount_excl_gst": w.get("amount_excl_gst"),
            }
            for w in stalled
        ],
        "by_status": breakdown(work_orders, "execution_status", "amount_excl_gst", top=8),
        "by_sector": breakdown(work_orders, "sector", "amount_excl_gst", top=8),
    }


def receivables_snapshot(work_orders: Sequence[Record]) -> dict:
    """Billing and collection health — where cash is stuck.

    Note the mixed GST bases in the source: billed/collected/receivable are
    recorded inclusive of GST while order value is reported exclusive, so the
    two are not directly comparable and are never subtracted from each other
    here.
    """
    receivable = [w for w in work_orders if (w.get("receivable") or 0) > 0]
    unbilled = [w for w in work_orders if (w.get("to_bill_excl_gst") or 0) > 0]
    priority = [w for w in work_orders if w.get("ar_priority")]
    negative = [w for w in work_orders if (w.get("to_bill_excl_gst") or 0) < 0]

    total_receivable = float(sum(w.get("receivable") or 0 for w in work_orders))
    total_unbilled = float(sum(w.get("to_bill_excl_gst") or 0 for w in unbilled))
    priority_value = float(sum(w.get("receivable") or 0 for w in priority))

    return {
        "basis_note": (
            "receivable/billed/collected are INCLUSIVE of GST in the source; "
            "order value is reported EXCLUSIVE. Do not net one against the other."
        ),
        "total_receivable_incl_gst": total_receivable,
        "total_receivable_incl_gst_display": format_inr(total_receivable),
        "accounts_with_receivable": len(receivable),
        "total_unbilled_excl_gst": total_unbilled,
        "total_unbilled_excl_gst_display": format_inr(total_unbilled),
        "work_orders_unbilled": len(unbilled),
        "ar_priority_count": len(priority),
        "ar_priority_value": priority_value,
        "ar_priority_value_display": format_inr(priority_value),
        "top_receivables": [
            {
                "wo_id": w["wo_id"],
                "deal_name": w.get("deal_name"),
                "customer_code": w.get("customer_code"),
                "sector": w.get("sector"),
                "receivable": w.get("receivable"),
                "receivable_display": format_inr(w.get("receivable")),
                "invoice_status": w.get("invoice_status"),
                "ar_priority": w.get("ar_priority"),
            }
            for w in sorted(
                receivable, key=lambda x: x.get("receivable") or 0, reverse=True
            )[:5]
        ],
        "by_invoice_status": breakdown(work_orders, "invoice_status", "receivable"),
        "negative_to_bill_count": len(negative),
        "negative_to_bill_note": (
            f"{len(negative)} work order(s) show a NEGATIVE amount still to bill, "
            f"i.e. billed more than the order value — likely over-billing or a "
            f"data-entry error, worth checking"
        )
        if negative
        else None,
    }


# --------------------------------------------------------------------------
# Cross-board
# --------------------------------------------------------------------------

# The two boards have no shared entity key. Client codes live in disjoint
# namespaces (COMPANY### vs WOCOMPANY_###, zero overlap) and deal names are
# not unique — "Sakura" alone covers 27 deal rows and 9 work-order rows, so
# joining on name would produce 243 phantom pairs for that name and inflate
# every total derived from it.
JOIN_KEYS_AVAILABLE = ["owner", "sector"]
JOIN_KEYS_REFUSED = {
    "client_code": (
        "deals use COMPANY### and work orders use WOCOMPANY_###; the two "
        "namespaces have zero overlap, so no client can be matched across boards"
    ),
    "deal_name": (
        "deal names are not unique — one name can cover 27 deal rows and 9 "
        "work-order rows, so a name join multiplies rows instead of matching them"
    ),
}


class UnsafeJoinError(ValueError):
    """Raised when a caller tries to join the boards on a key that cannot work."""


def cross_board_view(
    deals: Sequence[Record], work_orders: Sequence[Record], by: str = "owner"
) -> dict:
    """The only safe cross-board comparison: aligned on owner or sector.

    Refuses any other key rather than returning a plausible-looking wrong
    answer. This is a deliberate constraint of the source data, not a
    limitation of the implementation.
    """
    if by in JOIN_KEYS_REFUSED:
        raise UnsafeJoinError(
            f"Cannot join the boards on '{by}': {JOIN_KEYS_REFUSED[by]}."
        )
    if by not in JOIN_KEYS_AVAILABLE:
        raise UnsafeJoinError(
            f"Cannot join the boards on '{by}'. Only {JOIN_KEYS_AVAILABLE} are "
            f"shared dimensions across both boards."
        )

    deal_side = {r[by]: r for r in breakdown(deals, by, "value_inr")}
    wo_side = {r[by]: r for r in breakdown(work_orders, by, "amount_excl_gst")}

    rows = []
    for key in sorted(set(deal_side) | set(wo_side)):
        d, w = deal_side.get(key, {}), wo_side.get(key, {})
        rows.append(
            {
                by: key,
                "deal_count": d.get("count", 0),
                "deal_value_sum": d.get("sum"),
                # The GST basis is baked into the quoted string rather than
                # left to a sibling field, because a model reading a bare
                # "₹4.82 Cr" reliably guesses the basis wrong.
                "deal_value_sum_display": _tagged(d.get("sum"), "deal value"),
                "work_order_count": w.get("count", 0),
                "work_order_value_sum": w.get("sum"),
                "work_order_value_sum_display": _tagged(w.get("sum"), "excl GST"),
                "present_on_both_boards": bool(d) and bool(w),
            }
        )

    only_deals = [r[by] for r in rows if not r["work_order_count"]]
    only_wo = [r[by] for r in rows if not r["deal_count"]]

    # Rank in code. Comparing "₹69.77 L" against "₹9.35 Cr" by eye is the same
    # order-of-magnitude trap as converting them, and a model reading a table
    # of mixed-unit strings gets "which is biggest" wrong. Answer it here.
    def _largest(key: str) -> str | None:
        scored = [r for r in rows if r.get(key) is not None]
        return max(scored, key=lambda r: r[key])[by] if scored else None

    return {
        "aligned_on": by,
        # Stating the population explicitly: without this the counts get
        # reported as "open pipeline", which they are not.
        "population": (
            "ALL deals at every stage (open, won, lost and on hold) and ALL "
            "work orders at every execution status. These are NOT open-pipeline "
            "figures — label them as totals across the whole board, or call "
            "analyze_deals with stage_groups=['open'] for pipeline only."
        ),
        "deal_value_basis": "deal value (INR), coverage is partial",
        "work_order_value_basis": "order value EXCLUSIVE of GST (INR)",
        "largest_by_deal_value": _largest("deal_value_sum"),
        "largest_by_work_order_value": _largest("work_order_value_sum"),
        "ranking_note": (
            "Use largest_by_* above rather than comparing the formatted "
            "strings yourself — ₹9.35 Cr is larger than ₹69.77 L, and that "
            "comparison is easy to get backwards."
        ),
        "rows": sorted(
            rows, key=lambda r: -(r.get("deal_value_sum") or 0)
        ),
        "caveats": [
            f"aligned on {by} only — the boards share no client or deal identifier, "
            f"so this compares totals per {by}, it does not match individual deals "
            f"to individual work orders",
            *(
                [f"{by}(s) present only on the deals board: {', '.join(map(str, only_deals))}"]
                if only_deals else []
            ),
            *(
                [f"{by}(s) present only on the work orders board: {', '.join(map(str, only_wo))}"]
                if only_wo else []
            ),
        ],
    }


# --------------------------------------------------------------------------
# Date helpers exposed to the agents
# --------------------------------------------------------------------------


def resolve_period(period: str, today: date | None = None) -> tuple[date, date, str]:
    """Turn a phrase like 'this quarter' into concrete bounds.

    Returns ``(start, end, description)``. The description is echoed back to
    the reader so a period is never silently assumed.
    """
    today = today or datetime.now().date()
    p = (period or "").strip().lower()

    if p in {"", "all", "all time", "everything"}:
        return date(1900, 1, 1), date(2999, 12, 31), "all time (no date filter)"

    if p in {"this quarter", "current quarter", "quarter"}:
        start, end = N.current_quarter(today)
        q = (today.month - 1) // 3 + 1
        return start, end, f"Q{q} {today.year} ({start} to {end})"

    if p in {"last quarter", "previous quarter"}:
        q = (today.month - 1) // 3 + 1
        year, q = (today.year - 1, 4) if q == 1 else (today.year, q - 1)
        start, end = N.quarter_bounds(year, q)
        return start, end, f"Q{q} {year} ({start} to {end})"

    if p in {"this month", "current month"}:
        start = today.replace(day=1)
        nxt = (
            date(start.year + 1, 1, 1)
            if start.month == 12
            else date(start.year, start.month + 1, 1)
        )
        return start, nxt - timedelta(days=1), f"{start:%B %Y}"

    if p in {"this year", "current year", "ytd", "year to date"}:
        return date(today.year, 1, 1), today, f"{today.year} year to date"

    if p in {"next quarter"}:
        q = (today.month - 1) // 3 + 1
        year, q = (today.year + 1, 1) if q == 4 else (today.year, q + 1)
        start, end = N.quarter_bounds(year, q)
        return start, end, f"Q{q} {year} ({start} to {end})"

    # Explicit "Q3 2026"
    parsed = N.parse_date(period)
    if isinstance(parsed.value, date):
        return parsed.value, parsed.value, f"{parsed.value}"

    return date(1900, 1, 1), date(2999, 12, 31), (
        f"could not interpret the period '{period}' — no date filter was applied"
    )
