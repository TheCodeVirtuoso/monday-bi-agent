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
import re
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

7. THERE IS NO HISTORY. Both boards are a single current snapshot, with no
   record of what they looked like last week or last month. So you cannot
   answer "is this getting better or worse", "is this new", or "how does this
   compare to last month" — and a founder's next question after any risk
   figure is usually exactly that. When your answer describes a problem, say
   plainly that the data cannot show whether it is growing or shrinking. What
   you CAN give instead is age: work orders carry an end date, so
   "days_past_end_date" tells you how long something has been late.

8. END WITH THE SO-WHAT. After the numbers, add one short line naming the
   single thing most worth doing about them — the specific account, owner or
   job to look at first. One line, concrete, no hedging. If the numbers
   genuinely imply no action, say that instead. This is the difference
   between reporting and business intelligence.

HOW TO WRITE

Lead with the direct answer in the first sentence — the number they asked
for, in bold. Then what it means. Then, only if it would change what they do,
the caveat.

Use a markdown table when you are reporting a breakdown across three or more
groups (by stage, by sector, by owner). Tables render properly, and a
five-row breakdown is unreadable as prose. Keep tables to the columns that
matter — usually the group, the count and the value.

Do NOT emit a section heading with nothing under it, and never write "not
requested", "N/A" or "no data" as a section's content. If a question is only
about pipeline, answer about pipeline and stop; do not stub out headings for
delivery and cash.

Keep it tight: a few short paragraphs, or a table plus two lines. No
preamble, no "great question", no closing offer of further help.
""".strip()

LEADERSHIP_SUFFIX = """

LEADERSHIP UPDATE MODE.

A leadership update is COMPLETE by definition. Before writing anything, call
BOTH analyze_deals AND analyze_work_orders in the SAME turn, whatever the
user's question was about. Ask the deals specialist for open pipeline value,
deal count and late-stage count; ask the work orders specialist for execution
status, anything stalled, and receivables including the largest debtor.

NEVER write "Not requested", "N/A", "no data" or any similar placeholder. If
you are tempted to, you have not called the specialist you needed. An update
with empty sections is worse than no update — a founder cannot paste it
anywhere.

Then produce a markdown block they can paste straight into Slack or email:

**Pipeline** — open value with its coverage, deal count, late-stage count
**Delivery** — active vs stalled work orders, and which ones are at risk
**Cash** — receivable outstanding, unbilled value, AR-priority accounts
**Watch list** — 2-3 bullets on what actually needs a decision this week
**Data caveats** — only those that would change a decision

The watch list is the part that earns the update. Numbers alone are a report;
the watch list is what someone should DO about them.
""".rstrip()


_MONEY_IN_TEXT = re.compile(r"₹\s?[\d,]+(?:\.\d+)?\s?(?:Cr|L|K)?", re.I)


def _normalise_figure(text: str) -> str:
    return re.sub(r"[\s,]", "", text).upper()


def collect_figures(payload) -> set[str]:
    """Every rupee figure the tools actually produced, normalised."""
    found: set[str] = set()
    if isinstance(payload, dict):
        for v in payload.values():
            found |= collect_figures(v)
    elif isinstance(payload, list):
        for v in payload:
            found |= collect_figures(v)
    elif isinstance(payload, str):
        for m in _MONEY_IN_TEXT.findall(payload):
            found.add(_normalise_figure(m))
    return found


def unsupported_figures(answer: str, allowed: set[str]) -> list[str]:
    """Rupee amounts in the answer that no tool produced.

    The whole promise of this system is that its numbers are computed, not
    generated. A prompt saying "quote the display string verbatim" is not
    enough on its own — a live leadership update rendered ₹73.25 Cr as
    ₹93.25 Cr, a single transposed digit that changes the headline by ₹20
    crore. Checking is cheap; trusting is not.
    """
    if not allowed:
        return []
    bad = []
    for raw in _MONEY_IN_TEXT.findall(answer or ""):
        if _normalise_figure(raw) not in allowed:
            bad.append(raw.strip())
    # Preserve order, drop duplicates.
    return list(dict.fromkeys(bad))


@dataclass
class Turn:
    """One question and everything that happened while answering it."""

    question: str
    answer: str = ""
    agent_results: list[dict] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None
    unverified_figures: list[str] = field(default_factory=list)
    # Raw tool output collected during the turn. Used to verify the answer's
    # figures and to build charts; never sent back to a model.
    raw_data: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "agents": self.agent_results,
            "tool_calls": self.tool_calls,
            "error": self.error,
            "unverified_figures": self.unverified_figures,
            "charts": self.charts,
        }

    @property
    def charts(self) -> list[dict]:
        """Breakdowns worth drawing, taken from what the tools returned.

        The agents answer in prose, but the breakdowns underneath are already
        structured — so the UI can plot the real numbers instead of asking a
        model to describe a distribution in words. Charting the tool output
        rather than the prose also means a chart cannot disagree with the
        data, however the answer is worded.
        """
        found: list[dict] = []
        seen: set[str] = set()

        def walk(node) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key.startswith("by_") and isinstance(value, list) and value:
                        field_name = key[3:]
                        rows = [
                            {
                                "label": str(r.get(field_name) or "(not recorded)"),
                                "value": r.get("sum") or 0,
                                "display": r.get("sum_display") or "",
                                "count": r.get("count", 0),
                            }
                            for r in value
                            if isinstance(r, dict)
                        ]
                        rows = [r for r in rows if r["value"] or r["count"]]
                        title = field_name.replace("_", " ")
                        if len(rows) >= 2 and title not in seen:
                            seen.add(title)
                            found.append({"title": title, "rows": rows[:8]})
                    else:
                        walk(value)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(self.raw_data)
        return found[:2]


class Orchestrator:
    """Holds one session's loaded boards and conversation history."""

    def __init__(self, client=None) -> None:
        # Raises LLMError with a message naming the exact env var to set.
        self.client = client or llm.get_client()
        self.boards: dict[str, BoardData] | None = None
        self.history: list[dict] = []
        # Raw tool output for the turn in flight — used to verify the figures
        # in the answer and to build charts. Never sent back to the model.
        self._turn_raw: list[dict] = []

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
            """Returns (id, payload_for_model, raw_data_for_us)."""
            if name in ("analyze_deals", "analyze_work_orders"):
                board = boards["deals" if name == "analyze_deals" else "work_orders"]
                build = (
                    AG.build_deals_agent if name == "analyze_deals"
                    else AG.build_work_orders_agent
                )
                result = await build(board, self.client).run(args["question"])
                return tool_id, result.to_dict(), result.raw_outputs
            if name == "check_data_consistency":
                report = A.integrity_check(
                    boards["deals"].records,
                    boards["work_orders"].records,
                    only_at_risk=args.get("only_at_risk", True),
                )
                return tool_id, report, [report]
            if name == "compare_boards":
                try:
                    view = A.cross_board_view(
                        boards["deals"].records,
                        boards["work_orders"].records,
                        by=args.get("by", "owner"),
                    )
                    return tool_id, view, [view]
                except A.UnsafeJoinError as exc:
                    return tool_id, {"error": str(exc)}, []
            return tool_id, {"error": f"unknown tool '{name}'"}, []

        done = await asyncio.gather(*(one(i, n, a) for i, n, a in calls))
        self._turn_raw.extend(r for _, _, raws in done for r in raws)
        return {tid: payload for tid, payload, _ in done}

    def _verify_figures(self, answer: str, turn: "Turn") -> str:
        """Flag any rupee figure the tools did not actually produce."""
        allowed: set[str] = set()
        for payload in self._turn_raw + turn.agent_results:
            allowed |= collect_figures(payload)

        bad = unsupported_figures(answer, allowed)
        if not bad:
            return answer

        turn.unverified_figures = bad
        listed = ", ".join(bad[:4])
        return (
            f"{answer}\n\n> ⚠️ **Unverified figure(s): {listed}.** These do not "
            f"match any value computed from the boards, so treat them as "
            f"suspect and ask again rather than quoting them."
        )

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
        self._turn_raw = []

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
                    turn.answer = self._verify_figures(response.text, turn)
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
            turn.raw_data = list(self._turn_raw)
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
