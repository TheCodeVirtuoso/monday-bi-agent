"""Loading and normalising the two boards.

One interface, two interchangeable backends:

* ``FileBackend``   — reads the source workbooks in ``data/``. Works today.
* ``MondayBackend`` — reads the same two boards over monday.com's GraphQL API.

Both emit the *same* normalized record shape, so nothing downstream changes
when you switch. Everything that could differ between them (column titles,
header offsets, junk rows) is absorbed here.

The normalized layer is also where every data-quality problem is detected and
counted. Agents receive records that are already clean plus an explicit list
of what was wrong — they never see a raw cell, and never have to guess.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

import config
import normalize as N

DATA_DIR = Path(__file__).parent / "data"

# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class BoardData:
    """Normalized rows from one board, plus everything that went wrong."""

    board: str
    records: list[dict]
    caveats: list[dict] = field(default_factory=list)
    rows_in_source: int = 0
    rows_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    source: str = "file"

    @property
    def summary(self) -> dict:
        return {
            "board": self.board,
            "source": self.source,
            "rows_in_source": self.rows_in_source,
            "rows_usable": len(self.records),
            "rows_dropped": self.rows_dropped,
            "drop_reasons": self.drop_reasons,
        }


class DataSourceError(RuntimeError):
    """Raised when a board cannot be read at all.

    Deliberately distinct from "the board was read and came back empty" —
    the orchestrator reports those two situations differently, because
    conflating them is how an agent ends up telling a founder they have no
    pipeline when really the API was down.
    """


# --------------------------------------------------------------------------
# Junk-row detection
# --------------------------------------------------------------------------


# Names that are not names. See the comment at the drop site in
# ``normalize_deals`` for where each of these comes from.
_PLACEHOLDER_NAMES = {
    "(unnamed)", "Task 1", "Task 2", "Task 3", "Item 1", "Item 2", "Item 3",
}


def _is_repeated_header(row: pd.Series, columns: list[str]) -> bool:
    """True if this row is a copy of the header pasted into the data.

    The deals tracker has two of these (source rows 50 and 179). They are not
    blank and they carry a plausible-looking Deal Name, so they survive every
    ordinary null filter — and then quietly appear as their own category in
    any breakdown. Detected by counting cells whose value equals their own
    column name.
    """
    hits = sum(1 for c in columns if str(row.get(c, "")).strip() == c)
    return hits >= 3


# --------------------------------------------------------------------------
# Deals
# --------------------------------------------------------------------------

# Which funnel bucket each stage belongs to. Derived once from the groupings
# in normalize so "is this deal still open?" has exactly one definition.
_STAGE_GROUP: dict[str, str] = {
    **{s: "open" for s in N.OPEN_DEAL_STAGES},
    **{s: "won" for s in N.WON_DEAL_STAGES},
    **{s: "lost" for s in N.LOST_DEAL_STAGES},
    **{s: "hold" for s in N.HOLD_DEAL_STAGES},
}

_EXEC_GROUP: dict[str, str] = {
    **{s: "active" for s in N.WO_ACTIVE_STATUSES},
    **{s: "stalled" for s in N.WO_STALLED_STATUSES},
    **{s: "done" for s in N.WO_DONE_STATUSES},
}


def normalize_deals(raw_rows: list[dict], source: str = "file") -> BoardData:
    """Turn raw deal rows into normalized records plus caveats."""
    cav = N.CaveatCollector(noun="deal")
    records: list[dict] = []
    dropped: dict[str, int] = {}

    for i, row in enumerate(raw_rows):
        name = N._clean(row.get("Deal Name"))

        # Placeholder names, from two different sources: "(unnamed)" is what
        # the importer writes for a source row with no deal name (monday
        # requires every item to have one), and "Task 1/2/3" are the sample
        # items monday auto-creates with a new board. Both are non-data and
        # must not survive into a count.
        if not name or name in _PLACEHOLDER_NAMES:
            dropped["no deal name"] = dropped.get("no deal name", 0) + 1
            continue
        if _is_repeated_header(pd.Series(row), list(row.keys())):
            dropped["repeated header row"] = dropped.get("repeated header row", 0) + 1
            continue

        stage = N.parse_deal_stage(row.get("Deal Stage"))
        cav.record("stage", stage.issue, name)

        status = N.parse_deal_status(row.get("Deal Status"))
        cav.record("status", status.issue, name)

        sector = N.parse_sector(row.get("Sector/service"))
        cav.record("sector", sector.issue, name)

        value = N.parse_money(row.get("Masked Deal value"))
        cav.record("deal value", value.issue, name)

        expected = N.parse_date(row.get("Tentative Close Date"))
        cav.record("expected close date", expected.issue, name)

        actual = N.parse_date(row.get("Close Date (A)"))
        # Actual close date is blank on 92% of rows by design (it is only
        # filled once a deal truly closes), so reporting every blank as a gap
        # would drown the real caveats. Only malformed values are recorded.
        if actual.issue and actual.issue != N.DATE_MISSING:
            cav.record("actual close date", actual.issue, name)

        created = N.parse_date(row.get("Created Date"))
        cav.record("created date", created.issue, name)

        prob = N.canonicalize(
            row.get("Closure Probability"), N.CANONICAL_CLOSURE_PROBABILITY
        )

        stage_value = str(stage.value) if stage.value else None
        records.append(
            {
                "deal_id": f"D{i:04d}",
                "deal_name": name,
                "owner": N._clean(row.get("Owner code")) or None,
                "client_code": N._clean(row.get("Client Code")) or None,
                "status": status.value,
                "stage": stage_value,
                "stage_letter": stage_value.split(".")[0] if stage_value and "." in stage_value else None,
                "stage_group": _STAGE_GROUP.get(stage_value or "", "unknown"),
                "probability": prob.value,
                "value_inr": value.value,
                "sector": sector.value,
                "product": N._clean(row.get("Product deal")) or None,
                "expected_close_date": _iso(expected.value),
                "actual_close_date": _iso(actual.value),
                "created_date": _iso(created.value),
            }
        )

    data = BoardData(
        board="deals",
        records=records,
        caveats=cav.to_dicts(),
        rows_in_source=len(raw_rows),
        rows_dropped=sum(dropped.values()),
        drop_reasons=dropped,
        source=source,
    )
    _add_structural_caveats(data)
    return data


def normalize_work_orders(raw_rows: list[dict], source: str = "file") -> BoardData:
    """Turn raw work-order rows into normalized records plus caveats."""
    cav = N.CaveatCollector(noun="work order")
    records: list[dict] = []
    dropped: dict[str, int] = {}

    for i, row in enumerate(raw_rows):
        serial = N._clean(row.get("Serial #"))
        name = N._clean(row.get("Deal name masked"))
        label = name or serial

        # A work order is identified by its Serial #, not its name — one row
        # legitimately has a blank deal name. That also catches monday's
        # placeholder items, which carry no serial, so no name check is
        # needed here; adding one would discard a real work order.
        if not serial:
            dropped["no serial number"] = dropped.get("no serial number", 0) + 1
            continue
        if _is_repeated_header(pd.Series(row), list(row.keys())):
            dropped["repeated header row"] = dropped.get("repeated header row", 0) + 1
            continue

        status = N.parse_execution_status(row.get("Execution Status"))
        cav.record("execution status", status.issue, label)

        sector = N.parse_sector(row.get("Sector"))
        cav.record("sector", sector.issue, label)

        nature = N.parse_nature_of_work(row.get("Nature of Work"))
        cav.record("nature of work", nature.issue, label)

        doc = N.parse_document_type(row.get("Document Type"))
        cav.record("document type", doc.issue, label)

        platform = N.parse_platform(
            row.get("Is any Skylark software platform part of the client deliverables in this deal?")
        )

        invoice_status = N.parse_invoice_status(row.get("Invoice Status"))
        billing_status = N.parse_billing_status(row.get("Billing Status"))

        # --- money -------------------------------------------------------
        # Excl-GST is the reporting basis (see README / Decision Log). Incl
        # is carried alongside so an invoice-matching question can still be
        # answered without re-reading the board.
        amount = N.parse_money(row.get("Amount in Rupees (Excl of GST) (Masked)"))
        cav.record("order value (excl GST)", amount.issue, label)
        amount_incl = N.parse_money(row.get("Amount in Rupees (Incl of GST) (Masked)"))

        billed = N.parse_money(row.get("Billed Value in Rupees (Excl of GST.) (Masked)"))
        billed_incl = N.parse_money(row.get("Billed Value in Rupees (Incl of GST.) (Masked)"))
        collected = N.parse_money(row.get("Collected Amount in Rupees (Incl of GST.) (Masked)"))
        to_bill = N.parse_money(row.get("Amount to be billed in Rs. (Exl. of GST) (Masked)"))
        cav.record("amount still to bill", to_bill.issue, label)
        to_bill_incl = N.parse_money(row.get("Amount to be billed in Rs. (Incl. of GST) (Masked)"))
        receivable = N.parse_money(row.get("Amount Receivable (Masked)"))
        cav.record("amount receivable", receivable.issue, label)

        # --- dates -------------------------------------------------------
        po_date = N.parse_date(row.get("Date of PO/LOI"))
        cav.record("PO/LOI date", po_date.issue, label)
        start = N.parse_date(row.get("Probable Start Date"))
        cav.record("probable start date", start.issue, label)
        end = N.parse_date(row.get("Probable End Date"))
        cav.record("probable end date", end.issue, label)
        delivery = N.parse_date(row.get("Data Delivery Date"))
        last_invoice = N.parse_date(row.get("Last invoice date"))

        # --- quantities --------------------------------------------------
        qty_po = N.parse_quantity(row.get("Quantities as per PO"))
        cav.record("PO quantity", qty_po.issue, label)
        qty_ops = N.parse_quantity(row.get("Quantity by Ops"))
        qty_billed = N.parse_quantity(row.get("Quantity billed (till date)"))
        qty_balance = N.parse_quantity(row.get("Balance in quantity"))

        status_value = str(status.value) if status.value else None
        records.append(
            {
                "wo_id": serial,
                "deal_name": name or None,
                "customer_code": N._clean(row.get("Customer Name Code")) or None,
                "owner": N._clean(row.get("BD/KAM Personnel code")) or None,
                "sector": sector.value,
                "type_of_work": N._clean(row.get("Type of Work")) or None,
                "nature_of_work": nature.value,
                "is_recurring": nature.value in N.RECURRING_NATURE_OF_WORK,
                "execution_status": status_value,
                "execution_group": _EXEC_GROUP.get(status_value or "", "unknown"),
                "document_type": doc.value,
                "platform": platform.value,
                "last_executed_month": N._clean(row.get("Last executed month of recurring project")) or None,
                # money
                "amount_excl_gst": amount.value,
                "amount_incl_gst": amount_incl.value,
                "billed_excl_gst": billed.value,
                "billed_incl_gst": billed_incl.value,
                "collected_incl_gst": collected.value,
                "to_bill_excl_gst": to_bill.value,
                "to_bill_incl_gst": to_bill_incl.value,
                "receivable": receivable.value,
                "ar_priority": N._clean(row.get("AR Priority account")).lower() == "priority",
                "invoice_status": invoice_status.value,
                "billing_status": billing_status.value,
                "wo_billing_state": N._clean(row.get("WO Status (billed)")) or None,
                "expected_billing_month": N._clean(row.get("Expected Billing Month")) or None,
                "actual_billing_month": N._clean(row.get("Actual Billing Month")) or None,
                # dates
                "po_date": _iso(po_date.value),
                "start_date": _iso(start.value),
                "end_date": _iso(end.value),
                "data_delivery_date": _iso(delivery.value),
                "last_invoice_date": _iso(last_invoice.value),
                # quantities — amount and unit stay paired; see caveats about
                # unitless rows before summing any of these.
                "qty_po": qty_po.value.amount,
                "qty_po_unit": qty_po.value.unit,
                "qty_ops": qty_ops.value.amount,
                "qty_billed": qty_billed.value.amount,
                "qty_balance": qty_balance.value.amount,
            }
        )

    data = BoardData(
        board="work_orders",
        records=records,
        caveats=cav.to_dicts(),
        rows_in_source=len(raw_rows),
        rows_dropped=sum(dropped.values()),
        drop_reasons=dropped,
        source=source,
    )
    _add_structural_caveats(data)
    return data


def _iso(d: object) -> str | None:
    return d.isoformat() if isinstance(d, date) else None


# --------------------------------------------------------------------------
# Whole-board caveats
# --------------------------------------------------------------------------


def _add_structural_caveats(data: BoardData) -> None:
    """Add caveats about the board as a whole, not about individual cells.

    Per-row parsing catches bad values. It cannot catch problems that only
    show up in aggregate — a money column that is half empty, or a total
    dominated by two outliers. Those distort exactly the numbers a founder
    asks for, so they are detected here and travel with the data.
    """
    rows = data.records
    if not rows:
        return

    money_field = "value_inr" if data.board == "deals" else "amount_excl_gst"
    noun = "deal" if data.board == "deals" else "work order"
    date_field = "expected_close_date" if data.board == "deals" else "end_date"

    # Data recency. If the board's forward-looking dates all sit in the past,
    # every "this quarter" question returns nothing — and a bare zero reads as
    # "no business" when it actually means "this export is stale". Surfacing
    # the real range up front turns a dead end into a useful answer.
    dates = sorted(r[date_field] for r in rows if r.get(date_field))
    if dates:
        latest = date.fromisoformat(dates[-1])
        today = date.today()
        if latest < today:
            months = (today.year - latest.year) * 12 + today.month - latest.month
            data.caveats.insert(
                0,
                {
                    "field": date_field,
                    "issue": "data_recency",
                    "count": len(dates),
                    "examples": [],
                    "severity": "gap",
                    "text": (
                        f"this board's {date_field.replace('_', ' ')} values run "
                        f"{dates[0]} to {dates[-1]} — the most recent is ~{months} "
                        f"month(s) before today ({today}). Questions about the "
                        f"current quarter will correctly return nothing; say the "
                        f"data ends {dates[-1]} rather than reporting a bare zero"
                    ),
                },
            )

    values = [r[money_field] for r in rows if r.get(money_field) is not None]
    missing = len(rows) - len(values)

    if missing:
        pct = round(100 * missing / len(rows))
        data.caveats.insert(
            0,
            {
                "field": money_field,
                "issue": "value_coverage",
                "count": missing,
                "examples": [],
                "severity": "gap",
                "text": (
                    f"{missing} of {len(rows)} {noun}s ({pct}%) have no recorded "
                    f"value — every total below is computed on the {len(values)} "
                    f"that do, and understates the true figure"
                ),
            },
        )

    # Concentration: if a couple of rows dominate, the mean is not a
    # meaningful summary and the reader needs to know before quoting it.
    if len(values) >= 5:
        ordered = sorted(values, reverse=True)
        total = sum(ordered)
        if total > 0:
            top2_share = sum(ordered[:2]) / total
            if top2_share >= 0.25:
                data.caveats.insert(
                    0,
                    {
                        "field": money_field,
                        "issue": "value_concentration",
                        "count": 2,
                        "examples": [],
                        "severity": "note",
                        "text": (
                            f"value is highly concentrated — the 2 largest {noun}s "
                            f"are {top2_share:.0%} of the total "
                            f"(largest ₹{ordered[0]:,.0f} vs median ₹"
                            f"{ordered[len(ordered) // 2]:,.0f}); prefer the median "
                            f"and report the mean only with this caveat attached"
                        ),
                    },
                )


# --------------------------------------------------------------------------
# Backend: local files
# --------------------------------------------------------------------------


class FileBackend:
    """Reads the source workbooks shipped in ``data/``."""

    name = "file"

    # The work order tracker's real header is on the second row; the first is
    # an artifact of the original export and is entirely empty.
    SPECS = {
        "deals": ("deal_tracker.xlsx", 0),
        "work_orders": ("work_order_tracker.xlsx", 1),
    }

    def fetch(self, board: str) -> list[dict]:
        filename, skiprows = self.SPECS[board]
        path = DATA_DIR / filename
        if not path.exists():
            raise DataSourceError(
                f"Source workbook not found: {path}. Expected the trackers in "
                f"{DATA_DIR}."
            )
        try:
            df = pd.read_excel(path, skiprows=skiprows)
        except Exception as exc:  # pragma: no cover - depends on local file
            raise DataSourceError(f"Could not read {path.name}: {exc}") from exc
        df = df.dropna(how="all").dropna(axis=1, how="all")
        return df.to_dict("records")


# --------------------------------------------------------------------------
# Backend: monday.com GraphQL
# --------------------------------------------------------------------------

_ITEMS_QUERY = """
query ($boardId: ID!, $cursor: String) {
  boards(ids: [$boardId]) {
    name
    items_page(limit: 100, cursor: $cursor) {
      cursor
      items {
        id
        name
        column_values { id text value column { title } }
      }
    }
  }
}
"""


class MondayBackend:
    """Reads the same two boards from monday.com over GraphQL.

    Emits rows keyed by *column title* so the normalizers above are shared
    verbatim with :class:`FileBackend` — as long as the monday columns keep
    the titles they were imported with, switching backends changes nothing
    downstream.
    """

    name = "monday"

    BOARD_IDS = {
        "deals": config.DEALS_BOARD_ID,
        "work_orders": config.WORK_ORDERS_BOARD_ID,
    }

    def __init__(self, token: str | None = None) -> None:
        # Stripped again at the point of use, not only in config: a token with
        # a stray newline is rejected by httpx as an "Illegal header value",
        # an error that names neither the variable nor the whitespace. Cheap
        # insurance against a failure mode that is invisible in a dashboard.
        self.token = (token or config.MONDAY_API_TOKEN).strip()
        if not self.token:
            raise DataSourceError("MONDAY_API_TOKEN is not set.")

    async def fetch(self, board: str) -> list[dict]:
        board_id = self.BOARD_IDS.get(board)
        if not board_id:
            raise DataSourceError(f"No monday.com board id configured for '{board}'.")

        rows: list[dict] = []
        cursor: str | None = None
        headers = {
            "Authorization": self.token,
            "Content-Type": "application/json",
            "API-Version": "2024-10",
        }

        async with httpx.AsyncClient(timeout=config.MONDAY_TIMEOUT_SECONDS) as client:
            while True:
                payload = {
                    "query": _ITEMS_QUERY,
                    "variables": {"boardId": str(board_id), "cursor": cursor},
                }
                try:
                    resp = await client.post(
                        config.MONDAY_API_URL, json=payload, headers=headers
                    )
                except httpx.TimeoutException as exc:
                    raise DataSourceError(
                        f"monday.com timed out reading the {board} board."
                    ) from exc
                except httpx.HTTPError as exc:
                    raise DataSourceError(
                        f"Could not reach monday.com: {exc}"
                    ) from exc

                if resp.status_code == 401:
                    raise DataSourceError("monday.com rejected the API token (401).")
                if resp.status_code == 429:
                    raise DataSourceError("monday.com rate limit hit (429).")
                if resp.status_code >= 400:
                    raise DataSourceError(
                        f"monday.com returned HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                body = resp.json()
                if body.get("errors"):
                    msg = "; ".join(e.get("message", "?") for e in body["errors"])
                    raise DataSourceError(f"monday.com GraphQL error: {msg}")

                boards = (body.get("data") or {}).get("boards") or []
                if not boards:
                    raise DataSourceError(
                        f"monday.com returned no board for id {board_id}."
                    )

                page = boards[0]["items_page"]
                for item in page["items"]:
                    row: dict[str, Any] = {"__item_id": item["id"]}
                    # The board's own item name maps to whichever column the
                    # normalizer treats as the record's name.
                    row["Deal Name"] = item["name"]
                    row["Deal name masked"] = item["name"]
                    for cv in item["column_values"]:
                        title = (cv.get("column") or {}).get("title")
                        if title:
                            # Prefer the display text; fall back to the raw
                            # JSON value, which the parsers also understand.
                            row[title] = cv.get("text") or cv.get("value")
                    rows.append(row)

                cursor = page.get("cursor")
                if not cursor:
                    break

        return rows


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def _backend():
    if config.USE_MOCK_DATA:
        return FileBackend()
    return MondayBackend()


async def load_board(board: str) -> BoardData:
    """Fetch and normalize one board."""
    backend = _backend()
    if isinstance(backend, MondayBackend):
        raw = await backend.fetch(board)
    else:
        # File reads are blocking; keep the event loop free so the two boards
        # still load concurrently.
        raw = await asyncio.to_thread(backend.fetch, board)

    if board == "deals":
        return normalize_deals(raw, source=backend.name)
    return normalize_work_orders(raw, source=backend.name)


async def load_all() -> dict[str, BoardData]:
    """Fetch and normalize both boards concurrently.

    This is the only place fan-out happens. Both boards are independent
    network/disk reads, so they overlap; the agents above this layer reason
    over already-loaded data rather than each triggering their own fetch.
    """
    deals, work_orders = await asyncio.gather(
        load_board("deals"), load_board("work_orders")
    )
    return {"deals": deals, "work_orders": work_orders}
