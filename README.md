---
title: Monday BI Agent
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: gradio_app.py
pinned: false
short_description: BI agent over monday.com deals and work orders
---

# monday.com BI Agent

A conversational business-intelligence agent over two monday.com boards — a
**deals/pipeline tracker** and a **work-order delivery + billing tracker** —
for a drone-survey company. Ask it a question in plain English; it decides
which board(s) to consult, computes the numbers deterministically, and tells
you what the data cannot support instead of quietly rounding over it.

---

## Architecture

```
                        User (chat UI)
                              │
                              ▼
                    ┌───────────────────┐
                    │   Orchestrator    │  intent · scope · synthesis
                    └─────────┬─────────┘
                              │  fans out in ONE turn → asyncio.gather
              ┌───────────────┴───────────────┐
              ▼                               ▼
     ┌─────────────────┐             ┌──────────────────┐
     │  Deals Agent    │             │ Work Orders Agent│   judgment
     └────────┬────────┘             └────────┬─────────┘
              │                               │
              ▼                               ▼
     ┌──────────────────────────────────────────────────┐
     │  tools.py → analytics.py    (deterministic)      │   arithmetic
     └──────────────────────────┬───────────────────────┘
                                ▼
     ┌──────────────────────────────────────────────────┐
     │  data_source.py → normalize.py                   │   cleaning
     │  FileBackend (xlsx)  |  MondayBackend (GraphQL)   │
     └──────────────────────────────────────────────────┘
```

The organising principle is **agents own judgment, code owns transformation**.

Every date parse, sector mapping, status canonicalisation and rupee sum
happens in ordinary Python. The language models only decide *what to look at*
and *how to say it*. That matters for three reasons: the numbers are
reproducible run to run, the transformation logic is unit-tested, and the
model does not need to be large — which is why a free model runs this fine.

### Why two agents

Not because each board needs its own cleaning prompt — cleaning is code. It
is because each board has its own **meaning**. "Open" is stages A–F on the
deals board and a billing state on the work-order board; "sector" has 11
values on one and 6 on the other; money is GST-exclusive on one side and
GST-inclusive on the other. One generalist prompt holding both vocabularies
confuses them. Two specialists with narrow, explicit vocabularies do not.

They also genuinely run in parallel: when a question needs both boards, the
orchestrator emits both tool calls in a single turn and `asyncio.gather`
overlaps them, so two board analyses cost one wall-clock analysis.

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then add ONE llm key (see below)
uvicorn app:app --reload      # http://127.0.0.1:8000
```

The app runs against the workbooks in `data/` out of the box — no monday.com
account needed. The status line in the header tells you which data backend
and which model are live.

```bash
pytest                        # 87 tests, no network or API key required
python orchestrator.py "How's our pipeline looking?"   # CLI, needs a key
```

### Choosing an LLM provider

Three interchangeable backends. Set **one** key; the provider auto-detects.

| Provider | Cost | Notes |
|---|---|---|
| **Groq** (default) | Free tier | Fastest by a wide margin. `llama-3.3-70b-versatile`. Key: [console.groq.com](https://console.groq.com/keys) |
| **OpenRouter** | Free models available | Many models, one key. Free ids end in `:free` and change often. |
| **Anthropic** | Paid | Most reliable tool calling; useful fallback if a free model fumbles. |

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Switching providers is an env var, never a code change — `llm.py` translates
the neutral conversation and tool schemas to each provider's wire format.

## monday.com setup

### 1. Get an API token

monday.com → avatar (bottom-left) → **Developers** → **My Access Tokens** →
**Show**. Or go direct to
`https://<your-account>.monday.com/apps/manage/tokens`. Put it in `.env`:

```bash
MONDAY_API_TOKEN=eyJhbGciOi...
```

### 2. Provision the boards

```bash
python scripts/import_to_monday.py --dry-run   # check the column plan
python scripts/import_to_monday.py             # create + load (~16 min)
```

This creates **Deal Funnel** (346 items, 11 columns) and **Work Order
Tracker** (176 items, 33 columns), typed appropriately — `status` for
categoricals, `date` for dates, `numbers` for money, `text` for codes and
mixed-unit quantities.

It prints the two board ids when it finishes. Paste them into `.env` and flip
the backend:

```bash
MONDAY_DEALS_BOARD_ID=...
MONDAY_WORK_ORDERS_BOARD_ID=...
USE_MOCK_DATA=false
```

Three things worth knowing about the importer:

- **It does not clean anything.** The embedded header rows, the missing
  values, the original spellings and the mixed-unit quantity strings all
  survive the round trip — coping with them is what the agent is for. Only
  dates and numerics are interpreted, because a typed monday column requires
  it; anything unparseable is written empty rather than coerced, so "missing"
  and "malformed" stay distinguishable.
- **It is safe to re-run.** It deletes any board already using a target name
  first, so a part-finished run cannot leave a duplicate beside the real one.
- **It paces itself against monday's rate limit.** monday meters by
  *complexity*: 1,000,000 units per rolling 60s, and a `create_item` costs
  30,000 — a hard ceiling of ~33 items/min regardless of batching. The script
  reads the remaining budget from the `ratelimit` response header and waits
  only when the next batch would not fit. Expect ~16 minutes.

If you would rather import through monday's UI instead: **delete row 1 of the
work-order workbook first.** Its real header is on row 2, and monday's
importer will otherwise treat the blank first row as your column names.

### 3. Note on MCP vs API

The brief allows either. This uses the **GraphQL API**, because the deployed
service runs as its own process on its own host — it cannot borrow an MCP
connection configured in someone's editor. A token in the host's environment
is the only thing that works once this is not running on your laptop.

`MondayBackend` emits rows keyed by **column title**, which is why the
importer preserves the source headers verbatim: it lets the file backend and
the monday backend share one normalizer, and makes swapping between them a
config change rather than a code change.

---

## The data

| | Deals (`Deal tracker`) | Work Orders (`work order tracker`) |
|---|---|---|
| Source rows | 346 | 176 |
| Usable after cleaning | 342 | 176 |
| Columns | 12 | 38 (34 non-empty) |
| Currency | INR, masked | INR, masked, **GST-excl + GST-incl** |
| Header row | row 1 | **row 2** (row 1 is a blank export artifact) |

### What the cleaning layer handles

Detected from the real data, not assumed:

- **Two embedded header rows** (source rows 50 and 179) where every cell
  contains its own column name. They carry plausible deal names, survive
  every null filter, and would otherwise appear as their own category in
  every breakdown.
- **Deal stages encode funnel order** in a letter prefix, `A. Lead Generated`
  → `O. Not Relevant at all`. `Project Completed` has no prefix — an
  inconsistency preserved verbatim rather than silently renamed.
- **No "Energy" sector exists.** Energy work is recorded as `Renewables`
  and/or `Powerline`. Asking about "energy" expands to both, and the agent
  says so rather than inventing a category.
- **`Quantities as per PO` mixes ~20 units in one text column** — including
  three spellings of acre (`ACR`, `ACRE`, `ACERS`) — and 91 rows carry no
  unit at all.
- **`Billing Status` contains a typo** (`BIlled`) that would otherwise become
  its own bucket.
- Excel serials, `DD/MM` vs `MM/DD` ambiguity, `Q4 2026`, month-year strings,
  accounting-style negatives, `₹` and Indian digit grouping.

### What it refuses to do

**The two boards cannot be joined at the row level**, and the code enforces
this rather than trusting a prompt:

- Client codes are disjoint namespaces — `COMPANY089` vs `WOCOMPANY_002`,
  **zero overlap**.
- Deal names are **not unique**: `Sakura` covers 27 deal rows and 9
  work-order rows. A name join would produce 243 phantom pairs for that name
  alone and inflate every total derived from it.

`analytics.cross_board_view` accepts only `owner` or `sector` and raises
`UnsafeJoinError` — with the reason — for anything else.

### Caveats that travel with the data

Every aggregate reports its own coverage. `money_stats` returns
`count_with_value` and `coverage_pct` beside every sum, and warns when
coverage is under 90% or when the top two records dominate the total. Both
fire on this data:

- **52% of deals have no recorded value** (worse on open deals specifically),
  so every pipeline total is a floor, not a total.
- **The two largest deals are 46% of total recorded value**, so the mean is
  misleading and the median is the honest centre.

---

## Project layout

| File | Role |
|---|---|
| `normalize.py` | Pure parsers + canonical vocabularies. No I/O, no model. |
| `data_source.py` | Two backends → one normalized record shape + caveats. |
| `analytics.py` | Filters, statistics, snapshots, the join guard. |
| `tools.py` | Neutral tool schemas + deterministic dispatch. |
| `llm.py` | Provider adapter (Groq / OpenRouter / Anthropic). |
| `agents.py` | The two board specialists and their vocabularies. |
| `orchestrator.py` | Intent, parallel fan-out, synthesis. |
| `app.py` | FastAPI: `/`, `/api/health`, `/api/chat` (SSE), `/api/reset`. |
| `static/index.html` | Chat UI. No build step, no framework. |
| `tests/` | 87 tests. |

`/api/chat` streams progress events before the answer — a multi-agent turn
takes several seconds, and a blank box reads as a hang.

---

## Things to try

- *"How's our pipeline looking this quarter?"* — coverage caveat fires
- *"How are we doing in energy?"* — no such sector; expands and says so
- *"How much cash is outstanding, and who owes the most?"* — AR view
- *"What's stuck in delivery?"* — the stalled work orders, by name
- *"Match each deal to its work order"* — declines, and explains why
- Tick **leadership update** for a paste-ready Slack/email block

---

## Scope decisions

Deliberately not built, and why:

- **No write-back to monday.com.** Read-only by design.
- **No custom intent classifier.** The model's own understanding is enough;
  a hand-rolled one would be worse and slower.
- **No charts or slide generation.** Leadership updates are markdown. Chart
  rendering was deprioritised in favour of getting the numbers right.
- **No database.** Boards load once per session and are cached in memory;
  a new session re-fetches.
- **No row-level cross-board join.** Not a limitation of the implementation
  — the source data cannot support one. See above.
