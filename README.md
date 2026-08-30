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

### Connecting the real monday.com boards

The deployed app authenticates to monday.com **itself** — it cannot use an
MCP connection from your editor, because it runs as a separate process on a
separate host. Set:

```bash
MONDAY_API_TOKEN=...              # monday.com → avatar → Developers → My access tokens
MONDAY_DEALS_BOARD_ID=...
MONDAY_WORK_ORDERS_BOARD_ID=...
USE_MOCK_DATA=false
```

`MondayBackend` emits rows keyed by **column title**, so as long as the
imported columns keep the titles in `data_source.py`, nothing downstream
changes. Import the two workbooks from `data/` as-is: do not pre-clean them.

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
