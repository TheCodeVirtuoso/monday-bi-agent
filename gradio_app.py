"""Gradio entry point, for Hugging Face Spaces.

This is a second *front end*, not a second agent — it drives the same
`Orchestrator` as the FastAPI service in ``app.py``. Nothing about intent,
fan-out, analytics or normalisation is duplicated here; this file only
handles chat plumbing and progress display.

Run locally:   python gradio_app.py
On Spaces:     README front matter sets sdk=gradio and app_file=gradio_app.py
"""

from __future__ import annotations

import asyncio

import gradio as gr

import config
import llm
from data_source import DataSourceError, load_all
from orchestrator import Orchestrator

# --------------------------------------------------------------------------
# Shared board cache
# --------------------------------------------------------------------------
# Boards are loaded once for the whole Space and shared by every visitor —
# they are read-only and identical for everyone, and re-fetching per session
# would hammer monday's rate limit. Conversation history is NOT shared: each
# browser session gets its own Orchestrator (see `new_session`), so two
# people using the Space at once cannot see each other's questions.

_boards = None
_boards_lock = asyncio.Lock()
_boards_error: str | None = None


async def get_boards():
    global _boards, _boards_error
    async with _boards_lock:
        if _boards is None and _boards_error is None:
            try:
                _boards = await load_all()
            except DataSourceError as exc:
                _boards_error = str(exc)
            except Exception as exc:
                _boards_error = f"{type(exc).__name__}: {exc}"
    return _boards


_sessions: dict[str, object] = {}


def session_for(session_hash: str):
    """One Orchestrator per browser session, so histories stay separate.

    Keyed off Gradio's per-connection ``session_hash`` rather than held in a
    ``gr.State``, because a State cannot appear in the examples list — and
    the examples are the fastest way for a reviewer to try the thing.
    """
    orch = _sessions.get(session_hash)
    if orch is None:
        try:
            orch = Orchestrator()
        except Exception as exc:
            orch = exc
        _sessions[session_hash] = orch
    return orch


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


async def respond(message: str, history, leadership: bool, request: gr.Request):
    """Stream progress, then the answer."""
    session = session_for(getattr(request, "session_hash", "default"))

    if isinstance(session, Exception):
        yield (
            f"**The agent is not configured.**\n\n{session}\n\n"
            f"Set the provider key in this Space's *Settings → Variables and "
            f"secrets*."
        )
        return

    boards = await get_boards()
    if boards is None:
        yield (
            f"**I could not read the board data, so I have nothing to report "
            f"rather than a guess.**\n\n{_boards_error}\n\nPlease retry."
        )
        return

    session.boards = boards  # reuse the shared load

    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event: dict) -> None:
        await queue.put(event)

    task = asyncio.create_task(
        session.ask(message, leadership=leadership, on_event=on_event)
    )

    # A multi-agent turn takes several seconds; without progress the box
    # looks hung.
    steps: list[str] = []
    while not task.done() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        steps.append(event["detail"])
        yield "\n".join(f"*{s}…*" for s in steps[-3:])

    try:
        turn = await task
    except Exception as exc:
        yield f"Something went wrong answering that.\n\n`{type(exc).__name__}: {exc}`"
        return

    answer = turn.answer or "*(no answer produced)*"
    if turn.tool_calls:
        answer += f"\n\n<sub>via {', '.join(turn.tool_calls)}</sub>"
    yield answer


# --------------------------------------------------------------------------
# Status line
# --------------------------------------------------------------------------


def status_markdown() -> str:
    provider = llm.describe_provider()
    backend = "file (data/)" if config.USE_MOCK_DATA else "monday.com API"
    key = "connected" if provider.get("key_present") else (
        f"**missing `{provider.get('key_env_var')}`**"
    )
    return (
        f"**Data:** {backend} &nbsp;·&nbsp; "
        f"**Model:** `{provider.get('model')}` ({provider.get('provider')}) "
        f"&nbsp;·&nbsp; **Key:** {key}"
    )


EXAMPLES = [
    "How's our pipeline looking?",
    "What's stuck in delivery right now?",
    "How much cash is outstanding, and who owes the most?",
    "How are we doing in energy?",
    "Which owner has the strongest pipeline?",
    "Match each deal to its work order",
]

DESCRIPTION = """
Ask about the **deals pipeline**, **delivery status**, or **cash and
receivables** across two monday.com boards.

Numbers are computed deterministically in Python — the model chooses what to
look at and writes the answer, but never does the arithmetic. Totals come
with their coverage, and the agent says what the data *cannot* support rather
than rounding over it.
"""

# Gradio 6 moved `theme` from the Blocks constructor to launch().
with gr.Blocks(title="monday.com BI Agent") as demo:
    gr.Markdown("# monday.com BI Agent")
    gr.Markdown(DESCRIPTION)
    gr.Markdown(status_markdown())

    leadership = gr.Checkbox(
        label="Leadership update",
        info="Format the answer as a paste-ready Slack/email block",
        value=False,
    )

    # Gradio 6 dropped the `type` argument — the messages format is now the
    # only one — and caching examples would run every question at build time
    # against a live model, so it stays off.
    gr.ChatInterface(
        fn=respond,
        additional_inputs=[leadership],
        examples=[[q, False] for q in EXAMPLES],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(__import__("os").getenv("PORT", "7860")),
        theme=gr.themes.Soft(),
    )
