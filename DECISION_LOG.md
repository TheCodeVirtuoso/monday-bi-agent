# Decision Log

Design decisions, the reasoning behind them, and what I would do with more
time. Written against the real source data, not the plan I started with.

---

## 1. Agents own judgment; code owns transformation

**Decision.** The two domain agents do not clean data. Every date parse,
sector mapping, status canonicalisation and rupee sum happens in
deterministic Python (`normalize.py`, `analytics.py`). The models only decide
which board and filters a question needs, and how to phrase the result.

**Why.** The original plan had each domain agent clean its own board with a
specialist prompt. Three problems:

1. **Non-determinism.** An LLM parsing `03/04/2026` can produce different
   answers on different runs. The same question would return different
   numbers — the one failure mode you cannot recover from in a live demo.
2. **Latency.** Two extra model round-trips per question, for work that
   `difflib` and a regex do instantly.
3. **It is not the interesting part.** Date parsing is a solved problem.
   Deciding *what "open pipeline" means on this board* is not.

**Consequence.** Numbers are reproducible and unit-tested (87 tests, no
network). It also made the free-model migration in §6 cheap: the model never
touches arithmetic, so a smaller model is a much smaller risk than it would
otherwise be.

**Trade-off.** A value the code cannot map is reported as unmapped rather
than guessed. On this data that is the right call — see §3 — but it does mean
the vocabularies in `normalize.py` need extending when the source gains new
categories.

---

## 2. Two agents, not one, and not four

**Decision.** One specialist per board. No separate data-quality agent, no
separate report-writer agent.

**Why two.** Not because each board needs its own cleaning prompt — cleaning
is code. Because each board has its own **meaning**, and the vocabularies
actively collide:

| Term | Deals board | Work orders board |
|---|---|---|
| "open" | stages A–F | a billing state (`WO Status (billed)`) |
| "sector" | 11 values | 6 values |
| money | GST-**exclusive** | billed/collected are GST-**inclusive** |
| "completed" | `Project Completed` (a stage) | `Completed` (execution status) |

A single prompt holding both reliably conflates them. Two narrow prompts do
not.

**Why not more.** Caveats travel with the data that produced them, so a
data-quality agent would add a handoff and lose provenance. Formatting
happens once at the top, so a report-writer agent would add latency for
nothing. Fewer handoffs, fewer failure points.

**Parallelism is real, not decorative.** When a question needs both boards
the orchestrator emits both tool calls in one turn and `asyncio.gather`
overlaps them. Board loading is also concurrent, and cached per session — a
follow-up question costs model latency only.

---

## 3. Unmapped values are reported, never coerced

**Decision.** `canonicalize` tries exact match → alias table → containment →
fuzzy similarity, and if all four fail returns `None` with
`CATEGORY_UNMAPPED` rather than snapping to the nearest bucket.

**Why.** Coercion is how a category quietly ends up in the wrong total. The
validation run against the real data is the argument: of all distinct values
across nine categorical columns, the only three that failed to map were the
**embedded header rows** — which *should* fail. A more permissive matcher
would have silently filed `"Sector/service"` under a real sector.

**Related.** Ambiguity is surfaced rather than hidden. `03/04/2026` parses
day-first *and* returns `DATE_AMBIGUOUS`. A month-year like `Sep 2026` pins
to the 1st *and* says it assumed. The value is usable; the assumption is
visible.

---

## 4. Every aggregate carries its coverage

**Decision.** `money_stats` returns `count_with_value` and `coverage_pct`
beside every sum, and emits warnings when coverage drops below 90% or the top
two records exceed 25% of the total. Both agents are instructed never to
quote a total without its coverage.

**Why.** This data makes the point better than any argument:

- **52% of deals have no recorded value** (177 of 342), and it is worse on
  open deals specifically — only 55 of 140 carry a figure. A bare "pipeline
  is ₹73 Cr" is wrong by construction.
- **The two largest deals are 46% of total recorded value** (₹75.1 Cr against
  a ₹11 L median). The mean is not a summary of anything.

An agent that reports the sum and stops is confidently wrong. The
architectural fix is to make it *impossible* to receive a sum without its
coverage, rather than asking a prompt to remember.

---

## 5. Row-level cross-board joins are refused in code

**Decision.** `cross_board_view` accepts only `owner` or `sector` and raises
`UnsafeJoinError`, with the reason, for anything else. The orchestrator is
told to explain the limit rather than approximate around it.

**Why.** The original plan assumed a cross-board join was available. The data
says otherwise:

- Client codes are **disjoint namespaces** — `COMPANY089` vs
  `WOCOMPANY_002`, zero literal overlap across 199 and 51 distinct codes.
- Deal names overlap (52 of 58) but are **not unique**: `Sakura` covers 27
  deal rows and 9 work-order rows. Joining on name yields 243 phantom pairs
  for that one name.

The only shared namespace is owner code (`OWNER_001`–`006`, plus `007`
deals-only and `008` work-orders-only), and sector.

**Why in code rather than in the prompt.** A prompt instruction is advice; a
raised exception is a guarantee. This is the constraint most likely to
produce a confident, plausible, badly wrong answer, so it gets the strongest
enforcement available.

---

## 6. Provider-agnostic LLM layer, free model by default

**Decision.** `llm.py` exposes a neutral conversation and tool format and
translates to Groq, OpenRouter (both OpenAI-compatible) or Anthropic.
Switching is an env var. Groq is the default.

**Why it is affordable.** Because of §1. The model does no arithmetic, no
date parsing and no normalisation — it routes and it writes prose over
numbers handed to it. That is a low enough bar for a free 70B model. If the
model were doing the maths, I would not have made this change.

**What actually drove the choice.** Not cost — **tool-calling reliability**,
since the whole design hangs on function calling, and latency, which matters
more in a live demo than the bill does. Groq wins on both among free options;
Anthropic stays configured as a fallback for exactly the case where a free
model starts emitting malformed tool arguments.

**Defensive detail.** Malformed tool-call JSON degrades to an empty argument
dict rather than crashing the turn, and tool errors are returned *to the
model* as results so it can retry — both concessions to smaller models that a
frontier model would not need.

---

## 7. Errors never become silence

**Decision.** `DataSourceError` is distinct from an empty result. An
unreachable board says so; a genuinely empty selection says *that*. Provider
failures map to actionable messages naming the env var to fix.
`/api/health` performs a real load, and reports data health and model
credentials **separately**.

**Why.** Conflating "the API is down" with "you have no pipeline" is the
worst thing a BI tool can do. It is also the easy bug: both produce zero
rows.

---

## What I would do with more time

**Load-bearing, in priority order:**

1. **Evaluation harness.** A fixed set of questions with asserted numeric
   answers, run against each provider. Right now provider swaps are verified
   by translation unit tests and by hand; they should be verified by a suite
   that fails when a model regresses.
2. **Reconcile `Deal Status` against `Deal Stage`.** The board carries both
   (`Won/Dead/Open/On Hold` vs the A–O funnel). They are partly redundant and
   almost certainly disagree on some rows. I treat stage as authoritative and
   do not currently detect conflicts — those rows are a data-quality finding
   worth surfacing.
3. **Investigate the negative billing rows.** Six work orders show a negative
   amount still to bill (one at −₹82,907), meaning more was billed than the
   order was worth. Currently flagged as a caveat; it deserves a proper
   over-billing report.
4. **Trend over time.** Everything today is a snapshot. `Created Date` and
   `Date of PO/LOI` would support stage-velocity and win-rate-over-time
   questions, which is where "how are we doing?" usually leads next.

**Deliberately deferred:**

- **Charts and slide generation.** Leadership updates are markdown. Getting
  the numbers and caveats right was worth more than rendering them prettily.
- **Persistence.** No database; boards cache per session. A prototype does
  not need durable state, and adding it would have bought nothing
  demonstrable.
- **Write-back to monday.com.** Read-only by design.
- **A quantity-unit reconciliation pass.** `Quantities as per PO` has ~20
  units and 91 unitless rows. I normalise the units I can and refuse to sum
  across them; making quantity totals genuinely trustworthy is a data-entry
  fix, not a parsing one.
