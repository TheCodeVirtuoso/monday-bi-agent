"""Tool definitions and dispatch.

Two tiers:

* **Board tools** (``DEALS_TOOLS`` / ``WORK_ORDER_TOOLS``) are given to the two
  domain agents. They are thin, deterministic wrappers over ``analytics`` —
  the agent chooses *what* to ask for, the code computes it.
* **Orchestrator tools** (``ORCHESTRATOR_TOOLS``) delegate to the domain
  agents and expose the one cross-board comparison the data can support.

No tool here lets a model perform arithmetic. Filters go in, computed
statistics come out.

Schemas are declared in a neutral ``{name, description, parameters}`` shape;
``llm.py`` translates them for whichever provider is active. Descriptions are
written to be self-sufficient, because a smaller free model leans on them far
more heavily than a frontier one does.
"""

from __future__ import annotations

from typing import Any, Sequence

import analytics as A
import normalize as N
from data_source import BoardData

# --------------------------------------------------------------------------
# Shared schema fragments
# --------------------------------------------------------------------------

_SECTOR_ENUM = N.CANONICAL_SECTORS
_PERIOD_DESC = (
    "Time window in plain words: 'this quarter', 'last quarter', 'next quarter', "
    "'this month', 'this year', or 'all'. Omit for no date filter. Always tell "
    "the user which window was actually applied."
)

# --------------------------------------------------------------------------
# Deals tools
# --------------------------------------------------------------------------

DEALS_TOOLS: list[dict] = [
    {
        "name": "deals_query",
        "description": (
            "Filter the deals board and return computed statistics: counts, "
            "value sum/median/mean with coverage, and breakdowns by stage, "
            "sector and owner. Use this for any question about pipeline, "
            "deal value, win rates, or sales activity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sectors": {
                    "type": "array",
                    "items": {"type": "string", "enum": _SECTOR_ENUM},
                    "description": (
                        "Restrict to these sectors. Note there is no 'Energy' "
                        "sector in this data — energy work is recorded as "
                        "Renewables and/or Powerline."
                    ),
                },
                "stage_groups": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["open", "won", "lost", "hold"]},
                    "description": (
                        "'open' = stages A-F (live pipeline). 'won' = G-K and "
                        "Project Completed. 'lost' = L, N, O. 'hold' = M. "
                        "For pipeline questions use ['open'] only."
                    ),
                },
                "stages": {
                    "type": "array",
                    "items": {"type": "string", "enum": N.CANONICAL_DEAL_STAGES},
                    "description": "Restrict to specific named stages.",
                },
                "owners": {"type": "array", "items": {"type": "string"}},
                "period": {"type": "string", "description": _PERIOD_DESC},
                "date_field": {
                    "type": "string",
                    "enum": ["expected_close_date", "created_date", "actual_close_date"],
                    "description": (
                        "Which date the period applies to. Default "
                        "'expected_close_date' for forward-looking pipeline "
                        "questions, 'created_date' for questions about new "
                        "business generated in a period."
                    ),
                },
            },
        },
    },
    {
        "name": "deals_snapshot",
        "description": (
            "Pre-built views of the whole deals board. 'pipeline' gives the "
            "open-pipeline picture with stage/sector/owner splits; 'funnel' "
            "gives won/lost counts and the win rate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["pipeline", "funnel"]},
            },
            "required": ["kind"],
        },
    },
]

# --------------------------------------------------------------------------
# Work order tools
# --------------------------------------------------------------------------

WORK_ORDER_TOOLS: list[dict] = [
    {
        "name": "work_orders_query",
        "description": (
            "Filter the work-orders board and return computed statistics: "
            "counts, order value with coverage, and breakdowns by execution "
            "status, sector and owner. Use for delivery, execution and "
            "project-status questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sectors": {
                    "type": "array",
                    "items": {"type": "string", "enum": _SECTOR_ENUM},
                },
                "execution_groups": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["active", "stalled", "done"]},
                    "description": (
                        "'active' = Ongoing / Executed until current month / "
                        "Partial Completed / Not Started. 'stalled' = "
                        "Pause-struck or Details pending from Client. "
                        "'done' = Completed."
                    ),
                },
                "nature_of_work": {
                    "type": "array",
                    "items": {"type": "string", "enum": N.CANONICAL_NATURE_OF_WORK},
                },
                "owners": {"type": "array", "items": {"type": "string"}},
                "recurring_only": {
                    "type": "boolean",
                    "description": "Restrict to Annual Rate and Monthly contracts.",
                },
                "period": {"type": "string", "description": _PERIOD_DESC},
                "date_field": {
                    "type": "string",
                    "enum": ["po_date", "start_date", "end_date", "last_invoice_date"],
                },
            },
        },
    },
    {
        "name": "work_orders_snapshot",
        "description": (
            "Pre-built views of the whole work-orders board. 'delivery' gives "
            "execution status including a list of every stalled job. "
            "'receivables' gives billing and collection health: total "
            "receivable, unbilled value, AR-priority accounts and the largest "
            "outstanding balances. Use 'receivables' for any cash, billing, "
            "invoice or collection question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["delivery", "receivables"]},
            },
            "required": ["kind"],
        },
    },
]


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------


def compact(obj):
    """Shrink a tool result before it is serialised for the model.

    Tool results are the largest thing in every request, and on a free tier
    metered by tokens-per-minute that is the binding constraint — a single
    question was spending ~8,700 tokens against an 8,000/min ceiling.

    Two cheap wins, neither of which loses anything a reader would want:
    full-precision floats (``1132250.2480000001``) become integers at rupee
    magnitudes, and empty values are dropped. Nobody reports receivables to
    four decimal places.
    """
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return int(round(obj)) if abs(obj) >= 100 else round(obj, 2)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            v = compact(v)
            # Drop keys that carry no information, so the model reads less.
            if v is None or v == [] or v == {}:
                continue
            out[k] = v
        return out
    if isinstance(obj, (list, tuple)):
        return [compact(v) for v in obj]
    return obj


# How much detail a tool returns. Kept small deliberately: a breakdown with
# forty rows is not more useful to a founder than the top handful, and it
# costs tokens that the free tier meters by the minute.
TOP_GROUPS = 8
SAMPLE_ROWS = 4
MAX_CAVEATS = 6


def _caveat_texts(board: BoardData) -> list[str]:
    """Send caveats as sentences, not as the structs that produced them.

    The agent only ever quotes ``text``; ``field``/``issue``/``examples``
    exist for the code that groups them. Caveats are already ordered hard
    problems first, so truncating keeps the ones that matter.
    """
    return [c["text"] for c in board.caveats[:MAX_CAVEATS]]


def _apply_period(args: dict, default_date_field: str) -> tuple[dict, list[str]]:
    """Translate a period phrase into filter kwargs plus disclosure notes."""
    notes: list[str] = []
    kwargs: dict[str, Any] = {}
    period = args.get("period")
    if period and period.strip().lower() not in {"all", "all time", ""}:
        start, end, described = A.resolve_period(period)
        field = args.get("date_field") or default_date_field
        kwargs.update(date_field=field, date_from=start, date_to=end)
        notes.append(f"period read as {described}, applied to {field.replace('_', ' ')}")
    return kwargs, notes


def run_deals_tool(name: str, args: dict, board: BoardData) -> dict:
    records = board.records

    if name == "deals_snapshot":
        kind = args.get("kind")
        result = (
            A.pipeline_snapshot(records) if kind == "pipeline"
            else A.funnel_snapshot(records)
        )
        return compact({**result, "board_caveats": _caveat_texts(board)})

    if name == "deals_query":
        period_kwargs, notes = _apply_period(args, "expected_close_date")
        match: dict[str, Sequence[str]] = {}
        if args.get("stage_groups"):
            match["stage_group"] = args["stage_groups"]
        if args.get("stages"):
            match["stage"] = args["stages"]

        kept, filter_notes = A.filter_records(
            records,
            sectors=args.get("sectors"),
            owners=args.get("owners"),
            match=match or None,
            **period_kwargs,
        )
        return compact({
            "filters_applied": {
                k: v for k, v in args.items() if v not in (None, [], "")
            },
            "notes": notes + filter_notes,
            "matched_deals": len(kept),
            "value": A.money_stats(kept, "value_inr"),
            "by_stage": A.breakdown(kept, "stage", "value_inr", top=TOP_GROUPS),
            "by_sector": A.breakdown(kept, "sector", "value_inr", top=TOP_GROUPS),
            "by_owner": A.breakdown(kept, "owner", "value_inr", top=TOP_GROUPS),
            "largest_deals": [
                {
                    "deal_name": r["deal_name"],
                    "stage": r["stage"],
                    "sector": r["sector"],
                    "value": A.format_inr(r["value_inr"]),
                    "expected_close": r["expected_close_date"],
                    "owner": r["owner"],
                }
                for r in sorted(
                    kept, key=lambda x: x.get("value_inr") or 0, reverse=True
                )[:SAMPLE_ROWS]
            ],
        })

    raise ValueError(f"Unknown deals tool: {name}")


def run_work_order_tool(name: str, args: dict, board: BoardData) -> dict:
    records = board.records

    if name == "work_orders_snapshot":
        kind = args.get("kind")
        result = (
            A.delivery_snapshot(records) if kind == "delivery"
            else A.receivables_snapshot(records)
        )
        return compact({**result, "board_caveats": _caveat_texts(board)})

    if name == "work_orders_query":
        period_kwargs, notes = _apply_period(args, "start_date")
        match: dict[str, Sequence[str]] = {}
        if args.get("execution_groups"):
            match["execution_group"] = args["execution_groups"]
        if args.get("nature_of_work"):
            match["nature_of_work"] = args["nature_of_work"]
        if args.get("recurring_only"):
            match["nature_of_work"] = N.RECURRING_NATURE_OF_WORK

        kept, filter_notes = A.filter_records(
            records,
            sectors=args.get("sectors"),
            owners=args.get("owners"),
            match=match or None,
            **period_kwargs,
        )
        return compact({
            "filters_applied": {
                k: v for k, v in args.items() if v not in (None, [], "")
            },
            "notes": notes + filter_notes,
            "matched_work_orders": len(kept),
            "order_value_excl_gst": A.money_stats(kept, "amount_excl_gst"),
            "receivable_incl_gst": A.money_stats(kept, "receivable"),
            "by_execution_status": A.breakdown(
                kept, "execution_status", "amount_excl_gst", top=TOP_GROUPS
            ),
            "by_sector": A.breakdown(kept, "sector", "amount_excl_gst", top=TOP_GROUPS),
            "by_owner": A.breakdown(kept, "owner", "amount_excl_gst", top=TOP_GROUPS),
            "largest_work_orders": [
                {
                    "wo_id": r["wo_id"],
                    "deal_name": r["deal_name"],
                    "sector": r["sector"],
                    "execution_status": r["execution_status"],
                    "order_value": A.format_inr(r["amount_excl_gst"]),
                    "receivable": A.format_inr(r["receivable"]),
                    "end_date": r["end_date"],
                }
                for r in sorted(
                    kept, key=lambda x: x.get("amount_excl_gst") or 0, reverse=True
                )[:SAMPLE_ROWS]
            ],
        })

    raise ValueError(f"Unknown work order tool: {name}")


# --------------------------------------------------------------------------
# Orchestrator tools
# --------------------------------------------------------------------------

ORCHESTRATOR_TOOLS: list[dict] = [
    {
        "name": "analyze_deals",
        "description": (
            "Ask the Deals specialist about the deals/pipeline board (346 "
            "rows). Use for pipeline value, deal stages, win rates, sector "
            "mix of new business, owner performance on sales. Pass the "
            "specific sub-question you need answered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question for the deals specialist.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "analyze_work_orders",
        "description": (
            "Ask the Work Orders specialist about the delivery/billing board "
            "(176 rows). Use for execution status, stalled projects, billing, "
            "collections, receivables, AR priority and recurring contracts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The specific question for the work orders specialist.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "compare_boards",
        "description": (
            "Compare the two boards side by side, aligned on owner or sector. "
            "This is the ONLY safe cross-board view: the boards share no "
            "client or deal identifier, so individual deals cannot be matched "
            "to individual work orders. Attempting any other key returns an "
            "error explaining why."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "by": {
                    "type": "string",
                    "enum": ["owner", "sector"],
                    "description": "The shared dimension to align on.",
                },
            },
            "required": ["by"],
        },
    },
]
