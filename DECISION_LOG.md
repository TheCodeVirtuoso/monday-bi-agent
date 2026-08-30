# Decision Log

## 1. Key assumptions

**About the data.** The source is masked but internally consistent, and the
masking is not a defect to correct. I assumed no cross-board entity key
exists — verified, see §3. I assumed the letter prefixes on deal stages
(`A.` → `O.`) encode real funnel order and used them rather than inventing an
ordering. I assumed order value is GST-exclusive and
billed/collected/receivable are GST-inclusive, as the headers state, and
never net one against the other.

**About the business.** "Pipeline" means open deals only — stages A–F.
Stages G onward are won and belong to delivery; counting them as pipeline
double-counts revenue. There is no "Energy" sector in this data; energy work
is `Renewables` and `Powerline`, so the word expands to both and the agent
says so.

**About scope.** Read-only, per the brief. Board *setup* is separate: a
one-off script (`scripts/import_to_monday.py`) provisions the boards and
loads the workbooks. The agent itself never writes.

**Ambiguity I resolved without asking.** Where a question has two plausible
readings, the agent answers the likelier one and states the assumption in one
line, rather than stalling on a clarifying question. It asks only when no
reasonable default exists.

## 2. Architecture and trade-offs

**Agents own judgment; code owns transformation.** Every date parse, sector
mapping and rupee aggregation runs in deterministic Python. The models only
choose filters and write prose.

I originally planned LLM agents that cleaned their own board. I changed it
because an LLM parsing `03/04/2026` can answer differently on different runs
— the same question returning different numbers is the one failure you cannot
recover from in a demo. It also cost two model round-trips for work `difflib`
does instantly. *Trade-off:* the vocabularies in `normalize.py` must be
extended when the source gains new categories. Worth it for reproducibility
and 104 tests that run with no network.

**Two agents, not one and not four.** Not because each board needs its own
cleaning prompt — cleaning is code — but because the vocabularies collide:
"open" is a funnel stage on one board and a billing state on the other;
"completed" exists on both meaning different things; sector has 11 values on
one and 6 on the other. One prompt holding both conflates them. I skipped a
data-quality agent (caveats travel with the data that produced them) and a
report-writer agent (formatting happens once at the top) — each would add a
handoff and a failure point. When both boards are needed the orchestrator
emits both calls in one turn and `asyncio.gather` overlaps them.

**Unmapped values are reported, never coerced.** Of all distinct values
across nine categorical columns, the only three that failed to map were the
two embedded header rows — which *should* fail. A more permissive matcher
would have filed `"Sector/service"` under a real sector.

**Every aggregate carries its coverage.** 52% of deals have no recorded value
(worse on open deals: only 55 of 140), and the two largest deals are 46% of
total value. A bare sum is therefore wrong by construction. `money_stats`
returns `coverage_pct` beside every total and flags concentration, so it is
structurally impossible to receive a sum without its caveat — rather than
asking a prompt to remember.

**Free model by default, provider-agnostic.** Groq (`openai/gpt-oss-120b`),
switchable to OpenRouter or Anthropic by env var. This is only affordable
*because* of the first decision: the model does no arithmetic, so a small
model is a small risk. The deciding factor was tool-calling reliability, not
cost. *Discovered in testing:* Groq has retired the `llama-3.x` ids most
tutorials cite, and `qwen3.6-27b` does not reliably call tools at all.

**Money formatting is done in code.** The first live run reported ₹92.2 Cr as
"₹9.2 Cr" — my own prompt had invited the model to convert units inline. Both
formatting and ranking are now precomputed, so the model never converts or
compares magnitudes itself.

**Errors never become silence.** An unreachable board and an empty result are
distinct paths. A date filter matching nothing reports the range the data
*does* cover — the trackers end April 2026, so "this quarter" correctly
returns nothing, and a bare "₹0" would read as "no business".

## 3. The constraint worth naming

**The boards cannot be joined at row level, and the code enforces it.**
Client-code namespaces have zero overlap across 199 and 51 distinct codes.
Deal names overlap but are not unique: `Sakura` covers 27 deal rows and 9
work-order rows, so a name join yields 243 phantom pairs for that name alone.
Only owner and sector are shared.

`cross_board_view` raises `UnsafeJoinError` for any other key. A prompt
instruction is advice; an exception is a guarantee — and this is the
constraint most likely to produce a confident, plausible, badly wrong answer.

## 4. How I interpreted "leadership updates"

As a **paste-ready block, not a dashboard.** A founder wants something they
can drop into Slack or an email in the next thirty seconds, not another
surface to log into.

It is an on-demand mode (a toggle in the UI, `leadership=true` on the API),
not a separate agent — the data and caveats are identical, only the framing
changes. It emits five sections: **Pipeline** (open value with coverage),
**Delivery** (active vs stalled, named), **Cash** (receivable, unbilled, AR
priority), **Watch list** (2–3 items needing a decision this week), and **Data
caveats** (only those that would change a decision).

The deliberate choice is the watch list: numbers alone are not an update.
The explicit trade-off is **markdown only — no charts or slides.** With six
hours, getting the numbers and caveats right mattered more than rendering
them, and a wrong number in a pretty chart is worse than a right one in text.

## 5. With more time

1. **Evaluation harness** — fixed questions with asserted numeric answers, run
   per provider. Provider swaps are currently verified by translation unit
   tests and by hand; they should fail loudly when a model regresses.
2. **Reconcile `Deal Status` against `Deal Stage`** — the board carries both
   and they almost certainly disagree on some rows. I treat stage as
   authoritative and do not yet detect conflicts.
3. **Investigate negative billing rows** — six work orders show more billed
   than the order was worth (one at −₹82,907). Flagged as a caveat; deserves
   its own over-billing report.
4. **Trend over time** — everything today is a snapshot. `Created Date` would
   support stage velocity and win-rate-over-time, which is where "how are we
   doing?" usually leads next.

**Deliberately deferred:** write-back (read-only by design), a database
(session cache is enough for a prototype), charts, and trustworthy quantity
totals — `Quantities as per PO` mixes ~20 units with 91 unitless rows, which
is a data-entry fix, not a parsing one.
