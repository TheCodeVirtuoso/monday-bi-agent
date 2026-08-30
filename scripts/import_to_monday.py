"""One-off: create the two monday.com boards and load the source workbooks.

This is SETUP tooling, not part of the agent. The agent itself is read-only —
it never writes to monday.com. Run this once to provision the boards, then
put the printed board ids in your .env.

    python scripts/import_to_monday.py            # create and load
    python scripts/import_to_monday.py --dry-run  # show the plan only

Design note on "messy data": the assignment says the source is real-world
messy and the agent must cope. So this importer deliberately does NOT clean
anything. It preserves:

  * the two embedded header rows in the deals sheet
  * missing values as genuinely empty cells
  * every original spelling, casing and category variant
  * mixed-unit quantity strings ("5360 HA" next to a bare "4")

Only two things are interpreted, because a typed monday column requires it:
dates are written to date columns in ISO form, and numerics to number
columns. Anything that will not parse is written as empty rather than
coerced, which keeps "missing" and "malformed" distinguishable downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

API_URL = "https://api.monday.com/v2"
TOKEN = os.getenv("MONDAY_API_TOKEN", "")

DEALS_BOARD_NAME = "Deal Funnel"
WORK_ORDERS_BOARD_NAME = "Work Order Tracker"


# --------------------------------------------------------------------------
# Column plans
# --------------------------------------------------------------------------
# (source column, monday column title, monday column type)
#
# Titles are kept IDENTICAL to the source headers, because data_source.py
# keys rows by column title — that is what lets the file backend and the
# monday backend share one normalizer.

DEALS_COLUMNS = [
    ("Owner code",           "Owner code",           "text"),
    ("Client Code",          "Client Code",          "text"),
    ("Deal Status",          "Deal Status",          "status"),
    ("Close Date (A)",       "Close Date (A)",       "date"),
    ("Closure Probability",  "Closure Probability",  "status"),
    ("Masked Deal value",    "Masked Deal value",    "numbers"),
    ("Tentative Close Date", "Tentative Close Date", "date"),
    ("Deal Stage",           "Deal Stage",           "status"),
    ("Product deal",         "Product deal",         "text"),
    ("Sector/service",       "Sector/service",       "status"),
    ("Created Date",         "Created Date",         "date"),
]

WORK_ORDER_COLUMNS = [
    ("Customer Name Code",   "Customer Name Code",   "text"),
    ("Serial #",             "Serial #",             "text"),
    ("Nature of Work",       "Nature of Work",       "status"),
    ("Last executed month of recurring project", "Last executed month of recurring project", "text"),
    ("Execution Status",     "Execution Status",     "status"),
    ("Data Delivery Date",   "Data Delivery Date",   "date"),
    ("Date of PO/LOI",       "Date of PO/LOI",       "date"),
    ("Document Type",        "Document Type",        "status"),
    ("Probable Start Date",  "Probable Start Date",  "date"),
    ("Probable End Date",    "Probable End Date",    "date"),
    ("BD/KAM Personnel code", "BD/KAM Personnel code", "text"),
    ("Sector",               "Sector",               "status"),
    ("Type of Work",         "Type of Work",         "text"),
    ("Is any Skylark software platform part of the client deliverables in this deal?",
     "Skylark platform in deliverables?", "text"),
    ("Last invoice date",    "Last invoice date",    "date"),
    ("latest invoice no.",   "latest invoice no.",   "text"),
    ("Amount in Rupees (Excl of GST) (Masked)",       "Amount in Rupees (Excl of GST) (Masked)",       "numbers"),
    ("Amount in Rupees (Incl of GST) (Masked)",       "Amount in Rupees (Incl of GST) (Masked)",       "numbers"),
    ("Billed Value in Rupees (Excl of GST.) (Masked)", "Billed Value in Rupees (Excl of GST.) (Masked)", "numbers"),
    ("Billed Value in Rupees (Incl of GST.) (Masked)", "Billed Value in Rupees (Incl of GST.) (Masked)", "numbers"),
    ("Collected Amount in Rupees (Incl of GST.) (Masked)", "Collected Amount in Rupees (Incl of GST.) (Masked)", "numbers"),
    ("Amount to be billed in Rs. (Exl. of GST) (Masked)",  "Amount to be billed in Rs. (Exl. of GST) (Masked)",  "numbers"),
    ("Amount to be billed in Rs. (Incl. of GST) (Masked)", "Amount to be billed in Rs. (Incl. of GST) (Masked)", "numbers"),
    ("Amount Receivable (Masked)", "Amount Receivable (Masked)", "numbers"),
    ("AR Priority account",  "AR Priority account",  "text"),
    ("Quantity by Ops",      "Quantity by Ops",      "text"),
    ("Quantities as per PO", "Quantities as per PO", "text"),
    ("Quantity billed (till date)", "Quantity billed (till date)", "text"),
    ("Balance in quantity",  "Balance in quantity",  "text"),
    ("Invoice Status",       "Invoice Status",       "status"),
    # 'Expected Billing Month' exists in the source header but is empty on
    # every row, so it is not created here.
    ("Actual Billing Month", "Actual Billing Month", "text"),
    ("WO Status (billed)",   "WO Status (billed)",   "status"),
    ("Billing Status",       "Billing Status",       "status"),
]


# --------------------------------------------------------------------------
# API plumbing
# --------------------------------------------------------------------------


class MondayError(RuntimeError):
    pass


# monday meters the API by *complexity*, not request count: 1,000,000 units
# per rolling 60 seconds, and a create_item costs 30,000. That caps the import
# at ~33 items/minute no matter how the work is batched — batching 10 items
# into one request simply spends 300,000 at once.
#
# Rather than guess a sleep interval, we read the budget monday reports back
# on every response and wait only when the next batch would not fit.
COMPLEXITY_PER_ITEM = 30_000

_budget = {"remaining": 1_000_000, "resets_in": 60}


def _read_budget(resp: httpx.Response) -> None:
    """Parse: ratelimit: "complexityMinute";r=959960;t=58"""
    header = resp.headers.get("ratelimit", "")
    match = re.search(r'"complexityMinute";r=(\d+);t=(\d+)', header)
    if match:
        _budget["remaining"] = int(match.group(1))
        _budget["resets_in"] = int(match.group(2))


def await_budget(cost: int) -> None:
    """Block until `cost` complexity is available."""
    if _budget["remaining"] >= cost:
        return
    wait = _budget["resets_in"] + 2
    print(f"    complexity budget spent ({_budget['remaining']:,} left, "
          f"need {cost:,}); waiting {wait}s for the window to reset")
    time.sleep(wait)
    _budget["remaining"] = 1_000_000
    _budget["resets_in"] = 60


def gql(client: httpx.Client, query: str, variables: dict | None = None,
        attempts: int = 8) -> dict:
    """POST a GraphQL request, retrying on rate limits and 5xx."""
    for attempt in range(attempts):
        resp = client.post(
            API_URL,
            json={"query": query, "variables": variables or {}},
            headers={
                "Authorization": TOKEN,
                "Content-Type": "application/json",
                "API-Version": "2024-10",
            },
        )

        _read_budget(resp)

        if resp.status_code == 429 or resp.status_code >= 500:
            # On a complexity 429 the useful wait is until the window resets,
            # not an exponential guess.
            wait = _budget["resets_in"] + 2 if resp.status_code == 429 else 2 ** attempt
            print(f"    HTTP {resp.status_code}; waiting {wait}s")
            time.sleep(wait)
            _budget["remaining"] = 1_000_000
            continue

        if resp.status_code >= 400:
            raise MondayError(f"HTTP {resp.status_code}: {resp.text[:300]}")

        body = resp.json()
        errors = body.get("errors") or body.get("error_message")
        if errors:
            text = json.dumps(errors)
            # monday reports complexity exhaustion as a normal GraphQL error.
            if "complexity" in text.lower() or "rate" in text.lower():
                wait = 2 ** attempt
                print(f"    complexity/rate limit; backing off {wait}s")
                time.sleep(wait)
                continue
            raise MondayError(text[:400])
        return body["data"]

    raise MondayError("gave up after repeated rate limiting")


# --------------------------------------------------------------------------
# Value coercion
# --------------------------------------------------------------------------


def _blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if value != value:  # NaN / NaT
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def column_value(kind: str, value: object) -> object | None:
    """Translate one cell to monday's column_values format.

    Returns None to leave the cell genuinely empty — never a zero, never an
    empty-string label. Preserving "missing" as missing is the whole point;
    the agent reports coverage from it.
    """
    if _blank(value):
        return None

    if kind == "date":
        if isinstance(value, (datetime, pd.Timestamp)):
            return {"date": value.date().isoformat()}
        if isinstance(value, date):
            return {"date": value.isoformat()}
        try:
            parsed = pd.to_datetime(value, errors="coerce")
        except Exception:
            return None
        if parsed is None or pd.isna(parsed):
            return None
        return {"date": parsed.date().isoformat()}

    if kind == "numbers":
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number:
            return None
        return str(number)

    if kind == "status":
        # Labels are created on demand, so every original spelling and casing
        # variant survives the import — including the 'BIlled' typo.
        return {"label": str(value).strip()}

    return str(value).strip()


# --------------------------------------------------------------------------
# Board creation
# --------------------------------------------------------------------------

LIST_BOARDS = """
{ boards(limit: 100, state: active) { id name items_count } }
"""

DELETE_BOARD = """
mutation ($boardId: ID!) { delete_board(board_id: $boardId) { id } }
"""

CREATE_BOARD = """
mutation ($name: String!) {
  create_board(board_name: $name, board_kind: public) { id name }
}
"""


def clear_existing(client: httpx.Client, names: set[str]) -> None:
    """Delete any board already using one of our target names.

    A part-finished import leaves a board that looks real but is short some
    rows, and a second run would silently create a duplicate alongside it —
    which is exactly what happened once. Making the importer clean up first
    means re-running is always safe.
    """
    data = gql(client, LIST_BOARDS)
    stale = [b for b in data["boards"] if b["name"] in names]
    if not stale:
        return
    print(f"  found {len(stale)} existing board(s) with target names:")
    for board in stale:
        print(f"    deleting '{board['name']}' (id {board['id']}, "
              f"{board.get('items_count')} items)")
        gql(client, DELETE_BOARD, {"boardId": str(board["id"])})
        time.sleep(0.4)

CREATE_COLUMN = """
mutation ($boardId: ID!, $title: String!, $type: ColumnType!) {
  create_column(board_id: $boardId, title: $title, column_type: $type) { id title }
}
"""

def batch_mutation(size: int) -> str:
    """Build one mutation that creates ``size`` items via GraphQL aliases.

    monday has no bulk-create endpoint, and one HTTP request per item gets
    rate-limited hard on a small plan — the first attempt was backing off 8s
    at a time. Aliasing N create_item calls into a single request cuts the
    request count by a factor of N and stays well inside the limit.
    """
    args = ", ".join(f"$n{i}: String!, $v{i}: JSON!" for i in range(size))
    calls = "\n".join(
        f"  i{i}: create_item(board_id: $boardId, item_name: $n{i}, "
        f"column_values: $v{i}, create_labels_if_missing: true) {{ id }}"
        for i in range(size)
    )
    return f"mutation ($boardId: ID!, {args}) {{\n{calls}\n}}"


LIST_ITEMS = """
query ($boardId: ID!) {
  boards(ids: [$boardId]) { items_page(limit: 25) { items { id name } } }
}
"""

DELETE_ITEM = """
mutation ($itemId: ID!) { delete_item(item_id: $itemId) { id } }
"""

# monday seeds every new board with sample items. Left in place they become
# phantom records with all-null fields — a deal with no stage, no sector and
# no value, which is exactly the shape of a real data-quality problem and
# would be reported as one.
PLACEHOLDER_ITEMS = {"Task 1", "Task 2", "Task 3", "Item 1", "Item 2", "Item 3"}


def drop_placeholder_items(client: httpx.Client, board_id: str) -> None:
    data = gql(client, LIST_ITEMS, {"boardId": board_id})
    boards = data.get("boards") or []
    if not boards:
        return
    for item in boards[0]["items_page"]["items"]:
        if item["name"] in PLACEHOLDER_ITEMS:
            gql(client, DELETE_ITEM, {"itemId": str(item["id"])})
            print(f"    removed monday's placeholder item '{item['name']}'")
            time.sleep(0.3)


def build_board(client: httpx.Client, name: str, plan: list[tuple[str, str, str]]) -> tuple[str, dict]:
    data = gql(client, CREATE_BOARD, {"name": name})
    board_id = data["create_board"]["id"]
    print(f"  created board '{name}' -> id {board_id}")
    drop_placeholder_items(client, board_id)

    mapping: dict[str, tuple[str, str]] = {}
    for source_title, monday_title, kind in plan:
        created = gql(
            client, CREATE_COLUMN,
            {"boardId": board_id, "title": monday_title, "type": kind},
        )
        col_id = created["create_column"]["id"]
        mapping[source_title] = (col_id, kind)
        print(f"    + {kind:<8} {monday_title}")
        time.sleep(0.15)

    return board_id, mapping


BATCH_SIZE = 10


def load_rows(client: httpx.Client, board_id: str, mapping: dict,
              df: pd.DataFrame, name_column: str) -> int:
    total = len(df)
    created = 0

    # Pre-render every row so batching is a pure slicing problem.
    prepared: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        raw_name = row.get(name_column)
        # Rows with no name are still imported — the deals sheet has two, and
        # noticing them is part of the agent's job.
        item_name = "(unnamed)" if _blank(raw_name) else str(raw_name).strip()

        values = {}
        for source_title, (col_id, kind) in mapping.items():
            if source_title not in df.columns:
                continue
            v = column_value(kind, row.get(source_title))
            if v is not None:
                values[col_id] = v
        prepared.append((item_name[:255], json.dumps(values)))

    started = time.time()
    for start in range(0, total, BATCH_SIZE):
        chunk = prepared[start:start + BATCH_SIZE]
        variables: dict[str, object] = {"boardId": board_id}
        for i, (name, vals) in enumerate(chunk):
            variables[f"n{i}"] = name
            variables[f"v{i}"] = vals

        await_budget(len(chunk) * COMPLEXITY_PER_ITEM)
        gql(client, batch_mutation(len(chunk)), variables)
        created += len(chunk)

        rate = created / max(time.time() - started, 1) * 60
        eta = (total - created) / max(rate, 1)
        print(f"    {created}/{total} items  ({rate:.0f}/min, ~{eta:.1f} min left)")

    return created


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not TOKEN:
        print("MONDAY_API_TOKEN is not set in .env")
        return 1

    deals = pd.read_excel(ROOT / "data" / "deal_tracker.xlsx")
    work_orders = pd.read_excel(ROOT / "data" / "work_order_tracker.xlsx", skiprows=1)
    deals = deals.dropna(how="all").dropna(axis=1, how="all")
    work_orders = work_orders.dropna(how="all").dropna(axis=1, how="all")

    print(f"deals      : {len(deals)} rows x {len(DEALS_COLUMNS)} columns")
    print(f"work orders: {len(work_orders)} rows x {len(WORK_ORDER_COLUMNS)} columns")

    missing = [c for c, _, _ in DEALS_COLUMNS if c not in deals.columns]
    missing += [c for c, _, _ in WORK_ORDER_COLUMNS if c not in work_orders.columns]
    if missing:
        print("\nWARNING - source columns not found:", missing)

    if args.dry_run:
        print("\ndry run; nothing was created")
        return 0

    with httpx.Client(timeout=90) as client:
        print("\n=== cleaning up any previous run ===")
        clear_existing(client, {DEALS_BOARD_NAME, WORK_ORDERS_BOARD_NAME})

        print("\n=== Deal Funnel ===")
        deals_id, deals_map = build_board(client, DEALS_BOARD_NAME, DEALS_COLUMNS)
        n1 = load_rows(client, deals_id, deals_map, deals, "Deal Name")

        print("\n=== Work Order Tracker ===")
        wo_id, wo_map = build_board(client, WORK_ORDERS_BOARD_NAME, WORK_ORDER_COLUMNS)
        n2 = load_rows(client, wo_id, wo_map, work_orders, "Deal name masked")

    print("\n" + "=" * 66)
    print(f"imported {n1} deals and {n2} work orders")
    print("\nAdd these to your .env:\n")
    print(f"MONDAY_DEALS_BOARD_ID={deals_id}")
    print(f"MONDAY_WORK_ORDERS_BOARD_ID={wo_id}")
    print("USE_MOCK_DATA=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
