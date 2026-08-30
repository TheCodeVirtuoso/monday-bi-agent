"""The orchestrator: intent, fan-out, synthesis.

Flow for one question:

1. The orchestrator model decides which board(s) are in scope and what to ask
   each specialist. If the question is genuinely ambiguous it asks exactly one
   clarifying question instead of guessing.
2. Any specialist calls it issues in a single turn are executed CONCURRENTLY
   (``asyncio.gather``), so two board analyses cost one wall-clock analysis.
3. It synthesises one answer: the numbers, what they mean, and only the
   caveats that change the reader's conclusion.

Board data is fetched once per session and reused, so a follow-up question
costs model latency only — no re-read of the source.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date

import agents as AG
import analytics as A
import llm
import tools as T
from data_source import BoardData, DataSourceError, get_cached_boards

MAX_ORCHESTRATOR_ROUNDS = 4

ORCHESTRATOR_SYSTEM = """
You are a business-intelligence assistant for the founder of a drone-survey
company. You answer questions about two monday.com boards by delegating to
two specialists and combining what they return.

Today's date is {today}.

All money is Indian Rupees (INR). Money figures reach you ALREADY FORMATTED
in `*_display` fields (e.g. "₹92.22 Cr", "₹3.63 L"). Quote those strings
verbatim. NEVER convert a rupee figure into lakh or crore yourself — that
arithmetic is done in code precisely because doing it inline produces
order-of-magnitude errors. If no `_display` field exists, state the plain
rupee number exactly as given rather than abbreviating it.

YOUR TOOLS
- analyze_deals: the pipeline board (342 usable deals)
- analyze_work_orders: the delivery + billing board (176 work orders)
- compare_boards: the only safe cross-board view, aligned on owner or sector

HOW TO WORK

1. Decide scope. A pipeline/sales question goes to deals. A delivery,
   billing, cash or collections question goes to work orders. A question
   about the business overall goes to BOTH — and when you need both, issue
   both tool calls in the SAME turn so they run in parallel.

2. The boards CANNOT be joined at the row level. They share no client or deal
   identifier: client codes live in disjoint namespaces (COMPANY### vs
   WOCOMPANY_###) and deal names are not unique. If a question needs
   deal-to-work-order matching, say plainly that this data cannot support it
   and offer the owner- or sector-level comparison instead.

3. NEVER do arithmetic. Every figure must come from a specialist's findings
   verbatim. Do not add, subtract, average, or convert numbers yourself —
   including unit conversions. If you want a derived figure, ask a specialist
   for it. If two specialists each give you a total and you are tempted to add
   them, DO NOT: report them separately and say why they are not combined.

4. Terminology that matters:
   - "Pipeline" means OPEN deals, stages A-F. Won deals are delivery.
   - There is NO "Energy" sector. Energy work is Renewables + Powerline, and
     you must say so when the user uses the word.
   - Order value is EXCLUSIVE of GST; billed/collected/receivable figures are
     INCLUSIVE. Never mix the two bases.

5. Ambiguity: if the question has two materially different readings, pick the
   more likely one, ANSWER IT, and state the assumption in one line. Only ask
   a clarifying question when no reasonable default exists — and then ask
   exactly one.

6. Coverage is not optional. About half the deals carry no recorded value, so
   any pipeline total is a floor. Say that in the same breath as the number,
   not in a footnote.

HOW TO WRITE
Lead with the answer. Then the supporting numbers. Then, only if it changes
what the founder should do, the caveat. Keep it tight — a few short
paragraphs or a compact list, not an essay. No preamble, no "great
question", no closing offer of further help.
""".strip()

LEADERSHIP_SUFFIX = """

FORMAT: the user asked for a leadership update. Produce a markdown block they
can paste straight into Slack or an email:

**Pipeline** - open value with coverage, deal count, late-stage count
**Delivery** - active vs stalled work orders, anything at risk
**Cash** - receivable outstanding, unbilled value, AR-priority accounts
**Watch list** - 2-3 bullets on what actually needs a decision this week
**Data caveats** - only the ones that would change a decision

Use real figures from the specialists. No placeholders.
""".rstrip()


@dataclass
class Turn:
    """One question and everything that happened while answering it."""

    question: str
    answer: str = ""
    agent_results: list[dict] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "agents": self.agent_results,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }


class Orchestrator:
    """Holds one session's loaded boards and conversation history."""

    def __init__(self, client=None) -> None:
        # Raises LLMError with a message naming the exact env var to set.
        self.client = client or llm.get_client()
        self.boards: dict[str, BoardData] | None = None
        self.history: list[dict] = []

    # -- data ------------------------------------------------------------

    async def ensure_loaded(self) -> dict[str, BoardData]:
        """Attach the shared board data.

        Shared process-wide rather than per session: the boards are read-only
        and identical for everyone, so a copy per conversation wastes memory
        and re-fetches from monday for no benefit.
        """
        if self.boards is None:
            self.boards = await get_cached_boards()
        return self.boards

    def data_summary(self) -> dict:
        if not self.boards:
            return {}
        return {name: b.summary for name, b in self.boards.items()}

    # -- fan-out ---------------------------------------------------------

    async def fan_out(self, calls: list[tuple[str, str, dict]]) -> dict[str, dict]:
        """Execute the orchestrator's tool calls concurrently.

        ``calls`` is a list of ``(tool_use_id, tool_name, args)``. Specialist
        calls become real parallel LLM requests; ``compare_boards`` is
        deterministic and returns immediately.
        """
        boards = await self.ensure_loaded()

        async def one(tool_id: str, name: str, args: dict):
            if name == "analyze_deals":
                agent = AG.build_deals_agent(boards["deals"], self.client)
                return tool_id, (await agent.run(args["question"])).to_dict()
            if name == "analyze_work_orders":
                agent = AG.build_work_orders_agent(boards["work_orders"], self.client)
                return tool_id, (await agent.run(args["question"])).to_dict()
            if name == "compare_boards":
                try:
                    return tool_id, A.cross_board_view(
                        boards["deals"].records,
                        boards["work_orders"].records,
                        by=args.get("by", "owner"),
                    )
                except A.UnsafeJoinError as exc:
                    return tool_id, {"error": str(exc)}
            return tool_id, {"error": f"unknown tool '{name}'"}

        done = await asyncio.gather(*(one(i, n, a) for i, n, a in calls))
        return dict(done)

    # -- main loop -------------------------------------------------------

    async def ask(
        self,
        question: str,
        leadership: bool = False,
        on_event=None,
    ) -> Turn:
        """Answer one question.

        ``on_event`` is an optional async callback receiving progress
        dictionaries. A multi-agent turn takes several seconds; without
        progress the UI is a blank box and the user assumes it has hung.
        """
        turn = Turn(question=question)

        async def emit(kind: str, detail: str) -> None:
            if on_event:
                await on_event({"type": kind, "detail": detail})

        try:
            await emit("status", "Reading boards")
            await self.ensure_loaded()
        except DataSourceError as exc:
            # Never invent data when the source is unreachable.
            turn.error = str(exc)
            turn.answer = (
                f"I couldn't read the board data, so I have nothing to report "
                f"rather than a guess.\n\n**{exc}**\n\nPlease retry — if it "
                f"keeps failing, check the monday.com token and board IDs in "
                f"your `.env`."
            )
            return turn

        system = ORCHESTRATOR_SYSTEM.format(today=date.today().isoformat())
        if leadership:
            system += LEADERSHIP_SUFFIX

        messages = self.history + [llm.user(question)]

        for _ in range(MAX_ORCHESTRATOR_ROUNDS):
            await emit("status", "Working out what to look at")
            try:
                response = await self.client.complete(
                    system=system,
                    messages=messages,
                    tools=T.ORCHESTRATOR_TOOLS,
                    max_tokens=4000,
                )
            except llm.LLMError as exc:
                turn.error = str(exc)
                turn.answer = f"I couldn't reach the language model.\n\n**{exc}**"
                return turn
            except Exception as exc:
                turn.error = f"{type(exc).__name__}: {exc}"
                turn.answer = (
                    "I couldn't reach the language model to answer that. "
                    "Please retry."
                )
                return turn

            messages.append(llm.assistant(response))

            if not response.wants_tools:
                if response.text:
                    turn.answer = response.text
                    self.history = list(messages)
                    return turn
                # A turn with neither tool calls nor text is a dead end. Say so
                # rather than handing the user a blank bubble; do not persist
                # the empty turn into history either.
                turn.error = "model returned an empty response"
                turn.answer = (
                    "The model came back without an answer for that one. "
                    "Try rephrasing it, or ask for something more specific."
                )
                return turn

            calls = []
            for call in response.tool_calls:
                turn.tool_calls.append(call.name)
                calls.append((call.id, call.name, call.args))

            labels = {
                "analyze_deals": "Deals specialist",
                "analyze_work_orders": "Work orders specialist",
                "compare_boards": "Cross-board comparison",
            }
            running = " + ".join(labels.get(n, n) for _, n, _ in calls)
            await emit(
                "status",
                f"{running} running{' in parallel' if len(calls) > 1 else ''}",
            )

            outputs = await self.fan_out(calls)
            await emit("status", "Combining findings")

            for call in response.tool_calls:
                payload = outputs.get(call.id, {"error": "no result"})
                turn.agent_results.append(payload)
                messages.append(
                    llm.tool_result(call, json.dumps(payload, default=str))
                )

        turn.error = "exceeded tool rounds"
        turn.answer = (
            "I wasn't able to converge on an answer for that one. Try asking "
            "it in a more specific way."
        )
        return turn


# --------------------------------------------------------------------------
# CLI, for testing without the web layer
# --------------------------------------------------------------------------


async def _main() -> None:
    import sys

    orch = Orchestrator()
    boards = await orch.ensure_loaded()
    for name, b in boards.items():
        print(f"[{name}] {b.rows_in_source} rows -> {len(b.records)} usable "
              f"({b.rows_dropped} dropped) via {b.source}")

    questions = sys.argv[1:] or ["How's our pipeline looking this quarter?"]
    for q in questions:
        print("\n" + "=" * 78 + f"\nQ: {q}\n" + "-" * 78)
        turn = await orch.ask(q)
        print(turn.answer)
        if turn.tool_calls:
            print(f"\n[tools: {', '.join(turn.tool_calls)}]")


if __name__ == "__main__":
    asyncio.run(_main())
