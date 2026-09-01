"""The two domain agents.

Each owns one board and one vocabulary. What they do NOT do is transform
data: every date, sector, status and rupee figure they see has already been
normalized deterministically, and every statistic they quote was computed in
``analytics``. Their job is judgment — choosing the right filters for the
question, noticing what is odd about the result, and knowing which caveats
actually change the reader's conclusion.

That split is the reason two specialists are worth having. It is not that
each board needs its own cleaning prompt (cleaning is code); it is that each
board has its own *meaning* — "open" means stages A-F on one board and a
billing state on the other — and a single generalist prompt holding both
vocabularies at once reliably confuses them.

Both agents run concurrently; see ``orchestrator.fan_out``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import llm
import tools as T
from data_source import BoardData

MAX_TOOL_ROUNDS = 4

# --------------------------------------------------------------------------
# Shared instructions
# --------------------------------------------------------------------------

_SHARED_RULES = """
You are a specialist analyst for one board of a drone-survey company's
business trackers. All money is Indian Rupees (INR), masked but internally
consistent.

Hard rules:

1. NEVER do arithmetic yourself. Every number you report must come verbatim
   from a tool result. If you need a figure the tools did not give you, call
   another tool — do not estimate, and do not add numbers together in your
   head.

1a. NEVER convert between rupees, lakh and crore. Every money figure arrives
   with a matching pre-formatted `*_display` field (e.g. `sum_display`:
   "₹92.22 Cr"). QUOTE THAT STRING VERBATIM. Do not divide by 10,00,000 or
   1,00,00,000 yourself — that conversion is the single most common way to
   report a number that is wrong by a factor of ten. If a `_display` value is
   missing, give the plain rupee figure exactly as returned and do not
   abbreviate it.
2. ALWAYS report coverage with a total. The tools return `count_with_value`
   and `coverage_pct` alongside every sum. If coverage is below 90%, the sum
   is a floor, not a total, and you must say so in the same sentence.
3. Prefer the median to the mean when the tool warns about concentration.
4. Surface only the caveats that would change the reader's decision. Do not
   recite every data-quality note; pick the ones that matter for THIS
   question.
5. If the question cannot be answered from your board, say so plainly rather
   than answering a nearby question instead.
6. Be concise and concrete. You are writing for a founder who wants the
   number and the one thing they should notice about it.

7. There is NO history in this data — it is one current snapshot, with no
   record of previous weeks. Never imply a trend ("rising", "improving",
   "worse than last month"); you cannot know. Where something is late, report
   its AGE instead: work orders carry days_past_end_date, which is real.

Return your findings as prose. The orchestrator will combine yours with the
other specialist's, so do not write a greeting, a preamble, or a sign-off.
"""

DEALS_SYSTEM = _SHARED_RULES + """
YOUR BOARD: Deals / pipeline tracker (342 usable rows, 4 junk rows already
removed).

Vocabulary you must respect:

- Stages carry a letter prefix that encodes funnel order, A through O.
  A. Lead Generated -> B. Sales Qualified Leads -> C. Demo Done ->
  D. Feasibility -> E. Proposal/Commercials Sent -> F. Negotiations ->
  G. Project Won -> H. Work Order Received -> I. POC -> J. Invoice sent ->
  K. Amount Accrued. L is Lost, M is On Hold, N/O are Not Relevant.
  'Project Completed' has no letter prefix — an inconsistency in the source.
- OPEN PIPELINE MEANS STAGES A-F ONLY. Anything from G onward is already won
  and belongs to delivery. Counting won stages as pipeline double-counts
  revenue — this is the single easiest mistake to make on this board.
- Sectors in this data are: Mining, Renewables, Railways, Powerline,
  Construction, Manufacturing, Aviation, DSP, Tender, Security and
  Surveillance, Others. THERE IS NO "ENERGY" SECTOR. If the user says
  "energy", that maps to Renewables + Powerline and you must say so.

Known limits of this board, which you should raise when they bite:
- Only about half of all deals carry a recorded value, and coverage is worse
  on open deals specifically. Any pipeline total understates reality.
- Deal value is extremely concentrated; the two largest deals are ~46% of
  total recorded value.
- 'Close Date (A)' is blank on ~92% of rows by design — it fills in only on
  a real close. Use 'expected_close_date' (Tentative Close Date) for
  forward-looking questions.
"""

WORK_ORDERS_SYSTEM = _SHARED_RULES + """
YOUR BOARD: Work order / delivery + billing tracker (176 rows).

Vocabulary you must respect:

- Execution status groups: 'active' (Ongoing, Executed until current month,
  Partial Completed, Not Started), 'stalled' (Pause / struck, Details
  pending from Client), 'done' (Completed).
- Sectors on THIS board are only: Mining, Renewables, Railways, Powerline,
  Construction, Others. Five sectors that exist on the deals board never
  appear here.
- GST BASIS MATTERS. Order value is reported EXCLUSIVE of GST. Billed,
  collected and receivable figures are INCLUSIVE of GST. Never subtract one
  basis from the other, and always label which basis a figure is on.

Known limits of this board, which you should raise when they bite:
- 'Quantities as per PO' mixes ~20 different units in one column and 91 of
  the rows record no unit at all. Quantities may only be summed within a
  single unit, and any quantity total is unreliable.
- A handful of rows show a NEGATIVE amount still to bill, meaning more was
  billed than the order was worth. That is either over-billing or a data
  error and is worth flagging to a human.
- Some rows record a receivable with no matching invoice status.
"""


# --------------------------------------------------------------------------
# Agent runner
# --------------------------------------------------------------------------


@dataclass
class AgentResult:
    """What a domain agent hands back to the orchestrator."""

    agent: str
    findings: str
    tool_calls: list[dict] = field(default_factory=list)
    caveats: list[dict] = field(default_factory=list)
    error: str | None = None
    # Raw tool output, kept for the orchestrator to verify figures against and
    # for the UI to chart. Deliberately NOT part of to_dict(): the orchestrator
    # model reads the prose, and re-sending the underlying tables would double
    # the token cost of every turn for no gain.
    raw_outputs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "findings": self.findings,
            "tools_used": [c["name"] for c in self.tool_calls],
            "error": self.error,
        }


class DomainAgent:
    """A board specialist driving a small deterministic tool set."""

    def __init__(
        self,
        name: str,
        system: str,
        tool_schemas: list[dict],
        dispatch,
        board: BoardData,
        client,
    ) -> None:
        self.name = name
        self.system = system
        self.tool_schemas = tool_schemas
        self.dispatch = dispatch
        self.board = board
        self.client = client

    def _fail(self, calls: list[dict], error: str, raw: list[dict] | None = None) -> AgentResult:
        return AgentResult(
            agent=self.name,
            findings="",
            tool_calls=calls,
            caveats=self.board.caveats,
            error=error,
            raw_outputs=raw or [],
        )

    async def run(self, question: str) -> AgentResult:
        messages: list[dict] = [llm.user(question)]
        calls: list[dict] = []
        raw: list[dict] = []

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = await self.client.complete(
                    system=self.system,
                    messages=messages,
                    tools=self.tool_schemas,
                    max_tokens=3000,
                )
            except llm.LLMError as exc:
                return self._fail(calls, str(exc))
            except Exception as exc:
                return self._fail(calls, f"{type(exc).__name__}: {exc}")

            messages.append(llm.assistant(response))

            if not response.wants_tools:
                return AgentResult(
                    agent=self.name,
                    findings=response.text,
                    tool_calls=calls,
                    caveats=self.board.caveats,
                    raw_outputs=raw,
                )

            for call in response.tool_calls:
                calls.append({"name": call.name, "args": call.args})
                try:
                    output = self.dispatch(call.name, call.args, self.board)
                    raw.append(output)
                    content = json.dumps(output, default=str)
                except Exception as exc:
                    # Hand the error back as a tool result rather than
                    # aborting: a smaller model can often recover by calling
                    # the tool correctly on its next turn.
                    content = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    )
                messages.append(llm.tool_result(call, content))

        return self._fail(
            calls,
            f"stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer",
        )


def build_deals_agent(board: BoardData, client) -> DomainAgent:
    return DomainAgent(
        name="deals",
        system=DEALS_SYSTEM,
        tool_schemas=T.DEALS_TOOLS,
        dispatch=T.run_deals_tool,
        board=board,
        client=client,
    )


def build_work_orders_agent(board: BoardData, client) -> DomainAgent:
    return DomainAgent(
        name="work_orders",
        system=WORK_ORDERS_SYSTEM,
        tool_schemas=T.WORK_ORDER_TOOLS,
        dispatch=T.run_work_order_tool,
        board=board,
        client=client,
    )
