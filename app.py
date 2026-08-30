"""FastAPI service exposing the orchestrator over HTTP.

Endpoints
    GET  /               the chat UI
    GET  /api/health     board load status and which backend is in use
    POST /api/chat       ask a question; server-sent events stream progress
                         then the final answer
    POST /api/reset      clear a session's conversation history

Sessions are held in memory and keyed by a client-supplied id. That is the
right trade for a prototype: conversation history survives follow-up
questions within a browser session, and nothing is persisted anywhere.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import config
import llm
from data_source import DataSourceError, get_cached_boards
from orchestrator import Orchestrator

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="monday.com BI Agent", version="1.0")

# session id -> Orchestrator
_sessions: dict[str, Orchestrator] = {}
_sessions_lock = asyncio.Lock()


async def get_session(session_id: str) -> Orchestrator:
    async with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = Orchestrator()
        return _sessions[session_id]


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    leadership: bool = False


class ResetRequest(BaseModel):
    session_id: str = "default"
    refresh_data: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    """Report whether the boards can actually be read.

    Deliberately does the real load, so a green health check means the data
    path works — not merely that the process is up. It does NOT construct an
    Orchestrator, because the data path and the model credential are separate
    concerns: a missing API key should show up as its own warning, not as a
    failure to read the boards.
    """
    backend = "file" if config.USE_MOCK_DATA else "monday.com"
    result: dict = {"backend": backend, "llm": llm.describe_provider()}
    try:
        # Uses the shared cache: a health check is polled frequently, and
        # re-reading 522 items from monday on every poll would exhaust both
        # the instance's memory and monday's rate limit.
        boards = await get_cached_boards()
        result["status"] = "ok"
        result["boards"] = {name: b.summary for name, b in boards.items()}
    except DataSourceError as exc:
        result["status"] = "data_error"
        result["detail"] = str(exc)
    except Exception as exc:
        result["status"] = "error"
        result["detail"] = f"{type(exc).__name__}: {exc}"
    return result


@app.post("/api/reset")
async def reset(req: ResetRequest) -> dict:
    """Clear a session's history; optionally re-read the boards.

    ``refresh_data`` is the only way to pick up edits made in monday.com
    after the process started, since board data is cached for the life of the
    process.
    """
    async with _sessions_lock:
        orch = _sessions.get(req.session_id)
        if orch:
            orch.history = []
            orch.boards = None

    if req.refresh_data:
        try:
            await get_cached_boards(refresh=True)
        except DataSourceError as exc:
            return {"status": "data_error", "detail": str(exc)}

    return {"status": "ok", "data_refreshed": req.refresh_data}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Answer a question, streaming progress events as server-sent events."""

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(event: dict) -> None:
            await queue.put(event)

        try:
            orch = await get_session(req.session_id)
        except Exception as exc:
            # Most commonly a missing ANTHROPIC_API_KEY. Say so plainly
            # instead of failing the stream with a 500.
            yield "data: " + json.dumps(
                {
                    "type": "answer",
                    "answer": f"I can't answer questions yet.\n\n**{exc}**",
                    "tool_calls": [],
                    "error": str(exc),
                }
            ) + "\n\n"
            yield 'data: {"type": "done"}\n\n'
            return

        task = asyncio.create_task(
            orch.ask(req.message, leadership=req.leadership, on_event=on_event)
        )

        # Drain progress events until the answer is ready.
        while not task.done() or not queue.empty():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            yield f"data: {json.dumps(event)}\n\n"

        try:
            turn = await task
            payload = {
                "type": "answer",
                "answer": turn.answer,
                "tool_calls": turn.tool_calls,
                "error": turn.error,
                # Plotted from the tool output rather than the prose, so a
                # chart cannot disagree with the numbers behind it.
                "charts": turn.charts,
                "unverified_figures": turn.unverified_figures,
            }
        except Exception as exc:
            payload = {
                "type": "answer",
                "answer": (
                    "Something went wrong answering that. Please retry.\n\n"
                    f"`{type(exc).__name__}: {exc}`"
                ),
                "tool_calls": [],
                "error": str(exc),
            }

        yield f"data: {json.dumps(payload)}\n\n"
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
