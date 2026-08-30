"""Deterministic cleaning for messy monday.com column values.

Design principle for this project: **agents own judgment, code owns
transformation.** Every function here is pure and repeatable — the same raw
string always produces the same parsed value and the same issue code. That is
what makes the numbers in an answer reproducible run to run, which an LLM
doing the same parsing in a prompt could not guarantee.

Nothing here ever raises on bad input. A value that cannot be parsed comes
back as ``(None, "<issue code>")`` and the caller decides what to do with it —
in practice, keep the record but exclude it from calculations that need that
one field, and report it as a caveat.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, NamedTuple

# --------------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------------


class Parsed(NamedTuple):
    """A parsed value plus an optional issue code explaining what went wrong.

    ``issue`` is None on a clean parse. A value and an issue can both be
    present — that means we produced a usable value but had to make an
    assumption worth surfacing (e.g. an ambiguous DD/MM vs MM/DD date).
    """

    value: object | None
    issue: str | None = None


# Values that mean "the cell is empty", as opposed to "the cell has something
# in it that we could not understand". The distinction matters: missing data
# and malformed data are different stories to tell a founder.
MISSING_TOKENS = {
    "", "-", "--", "---", "n/a", "n.a.", "na", "none", "null", "nil",
    "tbd", "tba", "?", "??", "unknown", "#n/a", "#value!", "pending",
}


def _clean(raw: object) -> str:
    """Collapse a raw cell value to a trimmed, single-spaced string.

    Also absorbs pandas' two flavours of empty — ``NaN`` and ``NaT`` — which
    both satisfy ``raw != raw``. Without that check ``str(NaT)`` would arrive
    downstream as the literal text "NaT" and be reported as unparseable rather
    than missing.
    """
    if raw is None:
        return ""
    try:
        if raw != raw:  # NaN / NaT
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(raw)).strip()


def is_missing(raw: object) -> bool:
    return _clean(raw).lower() in MISSING_TOKENS


# --------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12,
    "december": 12,
}

# Excel stores dates as days since 1899-12-30 (the off-by-two leap year bug
# is already baked into that epoch).
_EXCEL_EPOCH = date(1899, 12, 30)

# When a date is genuinely ambiguous (both parts <= 12, e.g. "03/04/2024") we
# have to pick one. We pick day-first and *say so* rather than guessing
# silently — see DATE_AMBIGUOUS below.
DEFAULT_DAY_FIRST = True

DATE_MISSING = "date_missing"
DATE_UNPARSEABLE = "date_unparseable"
DATE_AMBIGUOUS = "date_ambiguous"
DATE_IMPLAUSIBLE = "date_implausible"


def _two_digit_year(y: int) -> int:
    """Expand a 2-digit year. 70-99 -> 19xx, 00-69 -> 20xx."""
    return 1900 + y if y >= 70 else 2000 + y


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_date(raw: object) -> Parsed:
    """Parse the date formats that actually turn up in exported CRM data.

    Handles ISO, slash/dash/dot separated numerics (with day-first vs
    month-first disambiguation), month names, bare month-years, quarters, and
    Excel serial numbers. Returns a ``date`` or an issue code.
    """
    # Excel and pandas hand us real date objects; take them directly rather
    # than round-tripping through a string. NaT must be tested first — it is
    # itself a datetime subclass, so an isinstance check alone would let it
    # through and yield a NaT-valued "successful" parse.
    if not _clean(raw):
        return Parsed(None, DATE_MISSING)
    if isinstance(raw, datetime):
        return Parsed(raw.date(), None)
    if isinstance(raw, date):
        return Parsed(raw, None)

    s = _clean(raw)
    if s.lower() in MISSING_TOKENS:
        return Parsed(None, DATE_MISSING)

    # monday.com's own date column JSON: {"date":"2024-03-15"}
    m = re.search(r'"date"\s*:\s*"(\d{4}-\d{2}-\d{2})"', s)
    if m:
        s = m.group(1)

    # Drop a trailing time component; we only ever aggregate by day.
    s = re.sub(r"[T ]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", "", s).strip()

    # --- ISO: 2024-03-15 ---
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        got = _safe_date(y, mo, d)
        return Parsed(got, None) if got else Parsed(None, DATE_UNPARSEABLE)

    # --- Excel serial number: 45372 ---
    if re.fullmatch(r"\d{4,5}(\.\d+)?", s):
        serial = float(s)
        if 20000 <= serial <= 60000:  # ~1954 to ~2064; outside that it is not a date
            return Parsed(_EXCEL_EPOCH + timedelta(days=int(serial)), None)
        return Parsed(None, DATE_IMPLAUSIBLE)

    # --- Quarter: Q1 2024 / 2024 Q1 / FY24 Q3 ---
    m = re.fullmatch(r"(?:fy)?\s*q([1-4])\s*[-/ ]?\s*(\d{2,4})", s, re.I) or re.fullmatch(
        r"(\d{4})\s*[-/ ]?\s*q([1-4])", s, re.I
    )
    if m:
        a, b = m.group(1), m.group(2)
        q, y = (int(a), int(b)) if int(a) <= 4 and len(a) == 1 else (int(b), int(a))
        y = _two_digit_year(y) if y < 100 else y
        return Parsed(date(y, 3 * (q - 1) + 1, 1), None)

    # --- Month name forms: "Jan 2024", "15 Jan 2024", "Jan 15, 2024" ---
    tokens = re.split(r"[\s,./-]+", s.lower())
    tokens = [t for t in tokens if t]
    month_tok = next((t for t in tokens if t.rstrip(".") in _MONTHS), None)
    if month_tok:
        mo = _MONTHS[month_tok.rstrip(".")]
        nums = [int(t) for t in tokens if t.isdigit()]
        years = [n for n in nums if n > 31]
        days = [n for n in nums if 1 <= n <= 31]
        if years:
            y = years[0] if years[0] > 99 else _two_digit_year(years[0])
        elif len(days) == 1:
            # "Jan 24" — a bare 2-digit number next to a month name reads as a
            # year far more often than as a day in this data.
            y, days = _two_digit_year(days[0]), []
        else:
            return Parsed(None, DATE_UNPARSEABLE)
        d = days[0] if days else 1
        got = _safe_date(y, mo, d)
        if not got:
            return Parsed(None, DATE_UNPARSEABLE)
        # A month-year with no day is a real value, but pinning it to the 1st
        # is our assumption, so flag it.
        return Parsed(got, None if days else DATE_AMBIGUOUS)

    # --- Numeric triples: 15/03/2024, 03-15-24, 15.03.2024 ---
    m = re.fullmatch(r"(\d{1,4})[/.\-](\d{1,2})[/.\-](\d{1,4})", s)
    if m:
        a, b, c = (int(g) for g in m.groups())
        # Year-first if the first part is unambiguously a year.
        if a > 31:
            got = _safe_date(a if a > 99 else _two_digit_year(a), b, c)
            return Parsed(got, None) if got else Parsed(None, DATE_UNPARSEABLE)

        y = c if c > 99 else _two_digit_year(c)
        issue = None
        if a > 12:            # first part must be the day
            d, mo = a, b
        elif b > 12:          # second part must be the day
            mo, d = a, b
        else:                 # both <= 12: genuinely ambiguous
            d, mo = (a, b) if DEFAULT_DAY_FIRST else (b, a)
            issue = DATE_AMBIGUOUS
        got = _safe_date(y, mo, d)
        return Parsed(got, issue) if got else Parsed(None, DATE_UNPARSEABLE)

    # --- Bare year: 2024 ---
    if re.fullmatch(r"(19|20)\d{2}", s):
        return Parsed(date(int(s), 1, 1), DATE_AMBIGUOUS)

    return Parsed(None, DATE_UNPARSEABLE)


# --------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------

MONEY_MISSING = "amount_missing"
MONEY_UNPARSEABLE = "amount_unparseable"
MONEY_NEGATIVE = "amount_negative"

_MULTIPLIERS = {"k": 1_000, "m": 1_000_000, "mm": 1_000_000, "bn": 1_000_000_000, "b": 1_000_000_000}


def parse_money(raw: object) -> Parsed:
    """Parse a currency-ish cell into a float.

    Copes with symbols, thousands separators, ``1.2M`` / ``450k`` shorthand,
    accounting-style ``(1,200)`` negatives, and stray currency codes.
    """
    s = _clean(raw)
    if s.lower() in MISSING_TOKENS:
        return Parsed(None, MONEY_MISSING)

    # monday.com numbers column can arrive as {"value": 1200}
    m = re.search(r'"value"\s*:\s*"?(-?[\d.]+)"?', s)
    if m:
        s = m.group(1)

    negative = bool(re.fullmatch(r"\(.*\)", s))
    s = s.strip("()")

    s = re.sub(r"(?i)\b(usd|eur|gbp|aud|cad|inr)\b", "", s)
    s = re.sub(r"[$€£₹¥,\s]", "", s)
    if s.startswith("-"):
        negative, s = True, s[1:]

    m = re.fullmatch(r"(\d*\.?\d+)\s*(k|mm|m|bn|b)?", s, re.I)
    if not m:
        return Parsed(None, MONEY_UNPARSEABLE)

    try:
        value = float(m.group(1))
    except ValueError:
        return Parsed(None, MONEY_UNPARSEABLE)

    if m.group(2):
        value *= _MULTIPLIERS[m.group(2).lower()]
    if negative:
        value = -value
        return Parsed(value, MONEY_NEGATIVE)
    return Parsed(value, None)


# --------------------------------------------------------------------------
# Percentages
# --------------------------------------------------------------------------

PCT_MISSING = "percent_missing"
PCT_UNPARSEABLE = "percent_unparseable"
PCT_ASSUMED_FRACTION = "percent_assumed_fraction"
PCT_OUT_OF_RANGE = "percent_out_of_range"


def parse_percent(raw: object) -> Parsed:
    """Parse a completion percentage, normalising to a 0-100 scale."""
    s = _clean(raw)
    if s.lower() in MISSING_TOKENS:
        return Parsed(None, PCT_MISSING)

    had_sign = "%" in s
    s = s.replace("%", "").strip()
    try:
        value = float(s)
    except ValueError:
        return Parsed(None, PCT_UNPARSEABLE)

    # "0.85" with no % sign almost certainly means 85%, not 0.85%.
    if not had_sign and 0 < value <= 1 and "." in s:
        return Parsed(value * 100, PCT_ASSUMED_FRACTION)
    if value < 0 or value > 100:
        return Parsed(value, PCT_OUT_OF_RANGE)
    return Parsed(value, None)


# --------------------------------------------------------------------------
# Categorical values (sector, status, stage)
# --------------------------------------------------------------------------

CATEGORY_MISSING = "category_missing"
CATEGORY_UNMAPPED = "category_unmapped"
CATEGORY_FUZZY = "category_fuzzy_matched"

# Words that carry no signal when matching a category label.
_NOISE_WORDS = {
    "sector", "sectors", "industry", "industries", "vertical", "verticals",
    "stage", "status", "the", "and", "of", "deal", "deals", "phase",
}

# --------------------------------------------------------------------------
# Canonical vocabularies
#
# Every list below was derived from the ACTUAL distinct values in the two
# source trackers, not invented. Counts in the comments are from the source
# profile and are there so a reader can tell a real category from a long tail.
# --------------------------------------------------------------------------

# Union of both boards. The deals tracker carries 11 real sectors; the work
# order tracker only ever uses 6 of them. That asymmetry is why a sector
# breakdown cannot be naively unioned across boards without saying which
# board it came from.
CANONICAL_SECTORS = [
    "Mining",                     # deals 106 / WO 100
    "Renewables",                 # deals 111 / WO  51
    "Railways",                   # deals  40 / WO  13
    "Powerline",                  # deals  26 / WO   6
    "Construction",               # deals   9 / WO   2
    "Manufacturing",              # deals   2 / WO   0
    "Aviation",                   # deals   1 / WO   0
    "DSP",                        # deals   7 / WO   0
    "Tender",                     # deals   5 / WO   0
    "Security and Surveillance",  # deals   1 / WO   0
    "Others",                     # deals  28 / WO   4
]

# Sectors that only ever appear on the deals board. Used to caveat any
# cross-board sector comparison.
DEALS_ONLY_SECTORS = [
    "Manufacturing", "Aviation", "DSP", "Tender", "Security and Surveillance",
]

# Aliases map how a *founder might phrase it* onto how the tracker records it.
# Keys are pre-normalised (lowercase, punctuation and noise words stripped),
# so "Energy Sector" and "energy" both arrive here as "energy".
SECTOR_ALIASES = {
    "coal": "Mining", "mine": "Mining", "mines": "Mining", "mineral": "Mining",
    "minerals": "Mining", "quarry": "Mining", "ore": "Mining",
    "solar": "Renewables", "wind": "Renewables", "green": "Renewables",
    "renewable": "Renewables", "clean": "Renewables", "hydro": "Renewables",
    "rail": "Railways", "railway": "Railways", "metro": "Railways",
    "train": "Railways", "rails": "Railways",
    "transmission": "Powerline", "grid": "Powerline", "tower": "Powerline",
    "t d": "Powerline", "transmission line": "Powerline", "lines": "Powerline",
    "civil": "Construction", "building": "Construction", "build": "Construction",
    "mfg": "Manufacturing", "factory": "Manufacturing",
    "industrial": "Manufacturing", "manufacture": "Manufacturing",
    "airport": "Aviation", "airline": "Aviation", "airports": "Aviation",
    "tenders": "Tender", "bid": "Tender", "bids": "Tender", "rfp": "Tender",
    "security": "Security and Surveillance",
    "surveillance": "Security and Surveillance",
    "other": "Others", "misc": "Others", "miscellaneous": "Others",
    "unclassified": "Others", "uncategorised": "Others",
}

# Some words a founder uses cover more than one recorded sector. Mapping them
# onto a single canonical value would silently drop data, so they expand to a
# LIST and the agent is told which sectors it actually covered.
SECTOR_GROUPS = {
    "energy": ["Renewables", "Powerline"],
    "power": ["Renewables", "Powerline"],
    "utilities": ["Renewables", "Powerline"],
    "infrastructure": ["Railways", "Powerline", "Construction"],
    "infra": ["Railways", "Powerline", "Construction"],
}

# --- Work orders: execution status -----------------------------------------
# Verbatim from the 'Execution Status' column.
CANONICAL_EXECUTION_STATUSES = [
    "Completed",                    # 117
    "Ongoing",                      #  25
    "Executed until current month", #  12
    "Not Started",                  #  11
    "Pause / struck",               #   4
    "Partial Completed",            #   2
    "Details pending from Client",  #   1
]

EXECUTION_STATUS_ALIASES = {
    "complete": "Completed", "done": "Completed", "finished": "Completed",
    "delivered": "Completed",
    "in progress": "Ongoing", "wip": "Ongoing", "active": "Ongoing",
    "running": "Ongoing", "underway": "Ongoing",
    "executed until current": "Executed until current month",
    "executed": "Executed until current month",
    "not started": "Not Started", "new": "Not Started", "yet to start": "Not Started",
    "pause struck": "Pause / struck", "paused": "Pause / struck",
    "struck": "Pause / struck", "stuck": "Pause / struck",
    "on hold": "Pause / struck", "halted": "Pause / struck",
    "partial completed": "Partial Completed", "partially complete": "Partial Completed",
    "details pending client": "Details pending from Client",
    "awaiting client": "Details pending from Client",
    "blocked": "Details pending from Client",
}

# Work still owed to a client, versus finished, versus stalled. Keeping these
# groupings in code means "what's active?" has one answer, not one per prompt.
WO_ACTIVE_STATUSES = [
    "Ongoing", "Executed until current month", "Partial Completed", "Not Started",
]
WO_STALLED_STATUSES = ["Pause / struck", "Details pending from Client"]
WO_DONE_STATUSES = ["Completed"]

# --- Deals: stage ----------------------------------------------------------
# The tracker already encodes funnel order in a letter prefix (A -> O), which
# is better than any ordering we could invent. 'Project Completed' is the one
# value with no prefix — an inconsistency in the source, preserved verbatim
# rather than silently renamed.
CANONICAL_DEAL_STAGES = [
    "A. Lead Generated",             # 74
    "B. Sales Qualified Leads",      # 14
    "C. Demo Done",                  #  9
    "D. Feasibility",                #  4
    "E. Proposal/Commercials Sent",  # 28
    "F. Negotiations",               # 13
    "G. Project Won",                # 27
    "H. Work Order Received",        # 46
    "I. POC",                        #  3
    "J. Invoice sent",               #  6
    "K. Amount Accrued",             #  2
    "L. Project Lost",               # 42
    "M. Projects On Hold",           # 20
    "N. Not relevant at the moment", # 19
    "O. Not Relevant at all",        # 18
    "Project Completed",             # 19  (no letter prefix in source)
]

DEAL_STAGE_ALIASES = {
    "lead": "A. Lead Generated", "leads": "A. Lead Generated",
    "lead generated": "A. Lead Generated", "new lead": "A. Lead Generated",
    "prospect": "A. Lead Generated", "inbound": "A. Lead Generated",
    "sql": "B. Sales Qualified Leads", "qualified": "B. Sales Qualified Leads",
    "sales qualified": "B. Sales Qualified Leads",
    "sales qualified leads": "B. Sales Qualified Leads",
    "demo": "C. Demo Done", "demo done": "C. Demo Done",
    "feasibility": "D. Feasibility",
    "proposal": "E. Proposal/Commercials Sent",
    "commercials": "E. Proposal/Commercials Sent",
    "proposal commercials sent": "E. Proposal/Commercials Sent",
    "quote": "E. Proposal/Commercials Sent",
    "quotation": "E. Proposal/Commercials Sent",
    "negotiation": "F. Negotiations", "negotiations": "F. Negotiations",
    "negotiating": "F. Negotiations",
    "won": "G. Project Won", "win": "G. Project Won",
    "project won": "G. Project Won", "closed won": "G. Project Won",
    "work order received": "H. Work Order Received",
    "wo received": "H. Work Order Received",
    "work order": "H. Work Order Received",
    "poc": "I. POC", "proof concept": "I. POC",
    "invoice sent": "J. Invoice sent", "invoiced": "J. Invoice sent",
    "amount accrued": "K. Amount Accrued", "accrued": "K. Amount Accrued",
    "lost": "L. Project Lost", "project lost": "L. Project Lost",
    "closed lost": "L. Project Lost", "dead": "L. Project Lost",
    "projects on hold": "M. Projects On Hold", "on hold": "M. Projects On Hold",
    "hold": "M. Projects On Hold",
    "not relevant at moment": "N. Not relevant at the moment",
    "not relevant now": "N. Not relevant at the moment",
    "not relevant at all": "O. Not Relevant at all",
    "project completed": "Project Completed", "completed": "Project Completed",
}

# Funnel groupings. 'Open pipeline' deliberately means stages A-F only —
# everything from G onward is already won and belongs to delivery, not
# pipeline. Counting H/J/K as pipeline would double-count won revenue.
OPEN_DEAL_STAGES = [
    "A. Lead Generated", "B. Sales Qualified Leads", "C. Demo Done",
    "D. Feasibility", "E. Proposal/Commercials Sent", "F. Negotiations",
]
WON_DEAL_STAGES = [
    "G. Project Won", "H. Work Order Received", "I. POC",
    "J. Invoice sent", "K. Amount Accrued", "Project Completed",
]
LOST_DEAL_STAGES = [
    "L. Project Lost", "N. Not relevant at the moment", "O. Not Relevant at all",
]
HOLD_DEAL_STAGES = ["M. Projects On Hold"]

# Late-stage = close enough to forecast on.
LATE_STAGE_DEALS = ["E. Proposal/Commercials Sent", "F. Negotiations"]

# --- Deals: status (a coarser rollup that sits alongside stage) -------------
CANONICAL_DEAL_STATUSES = ["Won", "Dead", "Open", "On Hold"]

DEAL_STATUS_ALIASES = {
    "win": "Won", "closed won": "Won", "closed": "Won",
    "lost": "Dead", "closed lost": "Dead", "dropped": "Dead",
    "active": "Open", "live": "Open", "in progress": "Open",
    "hold": "On Hold", "paused": "On Hold",
}

CANONICAL_CLOSURE_PROBABILITY = ["High", "Medium", "Low"]

# --- Work orders: commercial / billing vocabularies ------------------------
CANONICAL_INVOICE_STATUSES = [
    "Fully Billed", "Partially Billed", "Not billed yet", "Stuck",
]

INVOICE_STATUS_ALIASES = {
    "fully billed": "Fully Billed", "billed": "Fully Billed",
    "partially billed": "Partially Billed",
    # 'Billed- Visit 3' / 'Billed- Visit 7' are per-visit progress billing,
    # i.e. partial. Mapped explicitly so they are not read as fully billed.
    "billed visit 3": "Partially Billed", "billed visit 7": "Partially Billed",
    "not billed yet": "Not billed yet", "not billed": "Not billed yet",
    "unbilled": "Not billed yet", "pending": "Not billed yet",
    "stuck": "Stuck", "blocked": "Stuck",
}

# Note the source spells one of these 'BIlled' (capital I). Canonicalising on
# a lowercased key makes that typo a non-event rather than a separate bucket.
CANONICAL_BILLING_STATUSES = [
    "Billed", "Partially Billed", "Not Billable", "Update Required", "Stuck",
]

BILLING_STATUS_ALIASES = {
    "billed": "Billed", "fully billed": "Billed",
    "partially billed": "Partially Billed",
    "not billable": "Not Billable", "non billable": "Not Billable",
    "update required": "Update Required", "needs update": "Update Required",
    "stuck": "Stuck",
}

CANONICAL_WO_BILLING_STATE = ["Open", "Closed"]

CANONICAL_NATURE_OF_WORK = [
    "One time Project", "Proof of Concept", "Annual Rate Contract", "Monthly Contract",
]

NATURE_OF_WORK_ALIASES = {
    "one time": "One time Project", "one off": "One time Project",
    "onetime": "One time Project", "project": "One time Project",
    "poc": "Proof of Concept", "proof concept": "Proof of Concept",
    "pilot": "Proof of Concept",
    "arc": "Annual Rate Contract", "annual": "Annual Rate Contract",
    "annual contract": "Annual Rate Contract", "yearly": "Annual Rate Contract",
    "monthly": "Monthly Contract", "recurring": "Monthly Contract",
    "retainer": "Monthly Contract",
}

# Contracts that bill repeatedly — the recurring-revenue base.
RECURRING_NATURE_OF_WORK = ["Annual Rate Contract", "Monthly Contract"]

CANONICAL_DOCUMENT_TYPES = ["Purchase Order", "Email Confirmation", "LOA/LOI"]

DOCUMENT_TYPE_ALIASES = {
    "po": "Purchase Order", "purchase order": "Purchase Order",
    "email": "Email Confirmation", "email confirmation": "Email Confirmation",
    "loa": "LOA/LOI", "loi": "LOA/LOI", "loa loi": "LOA/LOI",
    "letter intent": "LOA/LOI", "letter award": "LOA/LOI",
}

# The 'Is any Skylark software platform...' column.
CANONICAL_PLATFORMS = ["NONE", "SPECTRA", "DMO", "SPECTRA + DMO"]

PLATFORM_ALIASES = {
    "none": "NONE", "no": "NONE", "nil": "NONE",
    "spectra": "SPECTRA", "dmo": "DMO", "spectra dmo": "SPECTRA + DMO",
}


def _normalize_label(s: str) -> str:
    """Lowercase, strip punctuation and category noise words."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    words = [w for w in s.split() if w and w not in _NOISE_WORDS]
    return " ".join(words)


def canonicalize(
    raw: object,
    canonical: Iterable[str],
    aliases: dict[str, str] | None = None,
    cutoff: float = 0.82,
) -> Parsed:
    """Map a free-text label onto a canonical value.

    Tries, in order: exact match, alias table, substring containment, then
    fuzzy string similarity. Anything that survives all four unmatched is
    returned as ``None`` with ``CATEGORY_UNMAPPED`` so it can be reported
    rather than silently bucketed into the wrong category.
    """
    aliases = aliases or {}
    canonical = list(canonical)

    s = _clean(raw)
    if s.lower() in MISSING_TOKENS:
        return Parsed(None, CATEGORY_MISSING)

    key = _normalize_label(s)
    if not key:
        return Parsed(None, CATEGORY_MISSING)

    by_canon = {_normalize_label(c): c for c in canonical}

    if key in by_canon:
        return Parsed(by_canon[key], None)
    if key in aliases:
        return Parsed(aliases[key], None)

    # Containment: "closed-won (signed 12 Mar)" -> "closed won"
    for norm, canon in by_canon.items():
        if norm and (norm in key or key in norm):
            return Parsed(canon, CATEGORY_FUZZY)
    for alias, canon in aliases.items():
        if alias and re.search(rf"\b{re.escape(alias)}\b", key):
            return Parsed(canon, CATEGORY_FUZZY)

    pool = list(by_canon) + list(aliases)
    match = difflib.get_close_matches(key, pool, n=1, cutoff=cutoff)
    if match:
        hit = match[0]
        return Parsed(by_canon.get(hit) or aliases[hit], CATEGORY_FUZZY)

    return Parsed(None, CATEGORY_UNMAPPED)


def parse_sector(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_SECTORS, SECTOR_ALIASES)


def parse_execution_status(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_EXECUTION_STATUSES, EXECUTION_STATUS_ALIASES)


def parse_deal_stage(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_DEAL_STAGES, DEAL_STAGE_ALIASES)


def parse_deal_status(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_DEAL_STATUSES, DEAL_STATUS_ALIASES)


def parse_invoice_status(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_INVOICE_STATUSES, INVOICE_STATUS_ALIASES)


def parse_billing_status(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_BILLING_STATUSES, BILLING_STATUS_ALIASES)


def parse_nature_of_work(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_NATURE_OF_WORK, NATURE_OF_WORK_ALIASES)


def parse_document_type(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_DOCUMENT_TYPES, DOCUMENT_TYPE_ALIASES)


def parse_platform(raw: object) -> Parsed:
    return canonicalize(raw, CANONICAL_PLATFORMS, PLATFORM_ALIASES)


def resolve_sector_query(term: object) -> tuple[list[str], str | None]:
    """Turn a user's sector word into the list of sectors it actually covers.

    Returns ``(sectors, note)``. A group word like "energy" expands to several
    recorded sectors and comes back with a note the agent must surface, so the
    reader knows the number spans Renewables *and* Powerline rather than some
    single column called "Energy" that does not exist in this data.
    """
    key = _normalize_label(_clean(term))
    if key in SECTOR_GROUPS:
        covered = SECTOR_GROUPS[key]
        return covered, (
            f"'{_clean(term)}' is not a sector in this data; read as "
            f"{' + '.join(covered)}"
        )
    parsed = parse_sector(term)
    if parsed.value:
        note = None
        if parsed.issue == CATEGORY_FUZZY:
            note = f"'{_clean(term)}' matched to sector '{parsed.value}' by similarity"
        return [str(parsed.value)], note
    return [], f"'{_clean(term)}' does not match any sector in this data"


# --------------------------------------------------------------------------
# Quantities
# --------------------------------------------------------------------------
# 'Quantities as per PO' mixes a number and its unit in one text cell
# ("5360 HA", "59.33", "4"), so the unit has to be split out before the
# numbers can be added up — and quantities in different units must never be
# summed together.

QTY_MISSING = "quantity_missing"
QTY_UNPARSEABLE = "quantity_unparseable"
QTY_NO_UNIT = "quantity_unit_missing"

# Observed units in 'Quantities as per PO', collapsed to canonical spellings.
# The source uses three spellings of acre (ACR / ACRE / ACERS, the last a
# typo) — folding them together is the difference between one acreage total
# and three unrelated ones.
_UNIT_CANON = {
    "ha": "HA", "hectare": "HA", "hectares": "HA",
    "acre": "ACRE", "acres": "ACRE", "acr": "ACRE", "acers": "ACRE",
    "km": "KM", "kms": "KM", "kilometre": "KM", "kilometres": "KM",
    "rkm": "RKM",  # route-km, distinct from plain km — do not merge
    "sqkm": "SQKM", "sq km": "SQKM", "km2": "SQKM",
    "nos": "NOS", "no": "NOS", "units": "NOS", "unit": "NOS", "each": "NOS",
    "visit": "VISIT", "visits": "VISIT",
    "month": "MONTH", "months": "MONTH", "mo": "MONTH",
    "day": "DAYS", "days": "DAYS",
    "tower": "TOWER", "towers": "TOWER",
    "site": "SITES", "sites": "SITES",
    "location": "LOCATION", "locations": "LOCATION",
    "mine": "MINES", "mines": "MINES",
    "rooftop": "ROOFTOPS", "rooftops": "ROOFTOPS",
    "pillar": "PILLARS", "pillars": "PILLARS",
    "image": "IMAGES", "images": "IMAGES",
    "subscription": "SUBSCRIPTIONS", "subscriptions": "SUBSCRIPTIONS",
    "mw": "MW", "au": "AU",
}

# Quantities may only be summed within a unit. Anything here is a bare number
# whose unit was never recorded, so it cannot join any of the above totals.
UNITLESS = None


class Quantity(NamedTuple):
    amount: float | None
    unit: str | None


def parse_quantity(raw: object) -> Parsed:
    """Split a quantity cell into an amount and a canonical unit.

    Returns a ``Quantity``. A bare number parses fine but comes back with
    ``QTY_NO_UNIT`` — the value is usable, but it cannot be safely added to a
    quantity that does carry a unit.
    """
    s = _clean(raw)
    if s.lower() in MISSING_TOKENS:
        return Parsed(Quantity(None, None), QTY_MISSING)

    m = re.fullmatch(r"([\d,]*\.?\d+)\s*([A-Za-z][A-Za-z\s\d]*)?", s)
    if not m:
        return Parsed(Quantity(None, None), QTY_UNPARSEABLE)

    try:
        amount = float(m.group(1).replace(",", ""))
    except ValueError:
        return Parsed(Quantity(None, None), QTY_UNPARSEABLE)

    raw_unit = (m.group(2) or "").strip().lower()
    if not raw_unit:
        return Parsed(Quantity(amount, None), QTY_NO_UNIT)

    unit = _UNIT_CANON.get(raw_unit, raw_unit.upper())
    return Parsed(Quantity(amount, unit), None)


# --------------------------------------------------------------------------
# Caveat collection
# --------------------------------------------------------------------------

# Human phrasing for each issue code. Keeping this here means the agents never
# have to invent wording for a data-quality problem.
_ISSUE_PHRASING = {
    DATE_MISSING: "no {field}",
    DATE_UNPARSEABLE: "an unreadable {field}",
    DATE_AMBIGUOUS: "an ambiguous {field} (assumed day-first / start of period)",
    DATE_IMPLAUSIBLE: "an implausible {field}",
    MONEY_MISSING: "no {field}",
    MONEY_UNPARSEABLE: "an unreadable {field}",
    MONEY_NEGATIVE: "a negative {field}",
    PCT_MISSING: "no {field}",
    PCT_UNPARSEABLE: "an unreadable {field}",
    PCT_ASSUMED_FRACTION: "a {field} recorded as a fraction (read as a percentage)",
    PCT_OUT_OF_RANGE: "a {field} outside 0-100",
    CATEGORY_MISSING: "no {field}",
    CATEGORY_UNMAPPED: "an unrecognised {field}",
    CATEGORY_FUZZY: "a {field} matched by similarity rather than exactly",
    QTY_MISSING: "no {field}",
    QTY_UNPARSEABLE: "an unreadable {field}",
    QTY_NO_UNIT: "a {field} with no unit recorded",
}

# Issues that are informational rather than a real gap — we mapped the value
# successfully, we just want the reader to know how.
_SOFT_ISSUES = {
    CATEGORY_FUZZY, DATE_AMBIGUOUS, PCT_ASSUMED_FRACTION, MONEY_NEGATIVE,
    QTY_NO_UNIT,
}


@dataclass
class Caveat:
    field: str
    issue: str
    count: int
    examples: list[str] = field(default_factory=list)
    soft: bool = False

    def to_text(self, noun: str) -> str:
        phrase = _ISSUE_PHRASING.get(self.issue, "a problem with {field}")
        phrase = phrase.format(field=self.field)
        plural = noun if self.count == 1 else f"{noun}s"
        text = f"{self.count} {plural} had {phrase}"
        if self.examples:
            shown = ", ".join(self.examples[:3])
            more = "" if self.count <= len(self.examples[:3]) else ", ..."
            text += f" (e.g. {shown}{more})"
        return text


class CaveatCollector:
    """Accumulates per-field parse issues and rolls them up for reporting.

    Grouping matters: forty rows with a missing close date should read as one
    sentence with a count, not forty bullet points.
    """

    def __init__(self, noun: str = "record") -> None:
        self.noun = noun
        self._buckets: dict[tuple[str, str], Caveat] = {}

    def record(self, field_name: str, issue: str | None, item_name: str = "") -> None:
        if not issue:
            return
        key = (field_name, issue)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = Caveat(
                field=field_name, issue=issue, count=0, soft=issue in _SOFT_ISSUES
            )
            self._buckets[key] = bucket
        bucket.count += 1
        # Names repeat heavily in this data (one deal name can cover 27 rows),
        # so de-duplicate examples — "e.g. Sakura, Sakura, Sakura" tells the
        # reader nothing.
        if item_name and len(bucket.examples) < 3 and item_name not in bucket.examples:
            bucket.examples.append(item_name)

    @property
    def caveats(self) -> list[Caveat]:
        # Hard problems first, then by how many rows they affect.
        return sorted(
            self._buckets.values(), key=lambda c: (c.soft, -c.count, c.field)
        )

    def to_text(self) -> list[str]:
        return [c.to_text(self.noun) for c in self.caveats]

    def to_dicts(self) -> list[dict]:
        return [
            {
                "field": c.field,
                "issue": c.issue,
                "count": c.count,
                "examples": c.examples,
                "severity": "note" if c.soft else "gap",
                "text": c.to_text(self.noun),
            }
            for c in self.caveats
        ]


# --------------------------------------------------------------------------
# Date helpers used by query filters
# --------------------------------------------------------------------------


def quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start_month = 3 * (quarter - 1) + 1
    start = date(year, start_month, 1)
    end = date(year + 1, 1, 1) if quarter == 4 else date(year, start_month + 3, 1)
    return start, end - timedelta(days=1)


def current_quarter(today: date | None = None) -> tuple[date, date]:
    today = today or datetime.now().date()
    return quarter_bounds(today.year, (today.month - 1) // 3 + 1)
