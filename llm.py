"""Provider-agnostic LLM layer.

The agents above this file never touch a vendor SDK. They build a neutral
conversation and a neutral tool list; this module translates both into
whichever provider is configured and translates the reply back.

Supported: Groq and OpenRouter (both OpenAI-compatible, one client) and
Anthropic (its own SDK). Switching is an env var, not a code change.

Why this is affordable here: all arithmetic, date parsing and normalisation
already happen in deterministic Python, so the model is only choosing filters
and writing prose over numbers it was handed. That is a low enough bar for a
small free model to clear — which would NOT be true if the model were doing
the maths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import config


# --------------------------------------------------------------------------
# Neutral types
# --------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(RuntimeError):
    """A provider call failed. Carries a message safe to show a user."""


# Neutral message shapes used by the agents:
#   {"role": "user",      "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
#   {"role": "tool",      "tool_call_id": str, "name": str, "content": str}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(response: LLMResponse) -> dict:
    return {
        "role": "assistant",
        "content": response.text,
        "tool_calls": response.tool_calls,
    }


def tool_result(call: ToolCall, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.id,
        "name": call.name,
        "content": content,
    }


# --------------------------------------------------------------------------
# Tool schema translation
# --------------------------------------------------------------------------
# Tools are declared once, neutrally, as {name, description, parameters}.


def _tools_openai(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _tools_anthropic(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]


# --------------------------------------------------------------------------
# OpenAI-compatible backend (Groq, OpenRouter)
# --------------------------------------------------------------------------


class OpenAICompatibleClient:
    """One client for every OpenAI-compatible provider."""

    def __init__(self, provider: dict) -> None:
        from openai import AsyncOpenAI

        headers = {}
        if provider["name"] == "openrouter":
            # OpenRouter asks callers to identify themselves; harmless
            # elsewhere and useful for its free-tier accounting.
            headers = {
                "HTTP-Referer": "https://github.com/monday-bi-agent",
                "X-Title": "monday.com BI Agent",
            }

        self.provider = provider
        self.model = provider["model"]
        self.client = AsyncOpenAI(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            default_headers=headers or None,
            timeout=90.0,
            max_retries=2,
        )

    @staticmethod
    def _messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.get("content") or None,
                }
                if m.get("tool_calls"):
                    entry["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.args),
                            },
                        }
                        for c in m["tool_calls"]
                    ]
                out.append(entry)
            elif m["role"] == "tool":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m["content"],
                    }
                )
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(system, messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = _tools_openai(tools)
            kwargs["tool_choice"] = "auto"

        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(_friendly_error(self.provider, exc)) from exc

        if not resp.choices:
            raise LLMError(f"{self.provider['name']} returned no choices.")

        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                # Smaller models occasionally emit malformed argument JSON.
                # Degrade to an empty call rather than crashing the turn —
                # the tool layer will then answer with its defaults.
                args = {}
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))

        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls)


# --------------------------------------------------------------------------
# Anthropic backend
# --------------------------------------------------------------------------


class AnthropicClient:
    def __init__(self, provider: dict) -> None:
        from anthropic import AsyncAnthropic

        self.provider = provider
        self.model = provider["model"]
        self.client = AsyncAnthropic(api_key=provider["api_key"], timeout=120.0)

    @staticmethod
    def _messages(messages: list[dict]) -> list[dict]:
        """Translate to Anthropic's block format.

        The one structural difference that matters: Anthropic expects every
        tool result for a single assistant turn inside ONE user message.
        Consecutive neutral 'tool' entries are therefore coalesced — splitting
        them would train the model out of making parallel tool calls.
        """
        out: list[dict] = []
        pending: list[dict] = []

        def flush() -> None:
            nonlocal pending
            if pending:
                out.append({"role": "user", "content": pending})
                pending = []

        for m in messages:
            if m["role"] == "tool":
                pending.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": m["tool_call_id"],
                        "content": m["content"],
                    }
                )
                continue

            flush()
            if m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls") or []:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": c.id,
                            "name": c.name,
                            "input": c.args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks or " "})
            else:
                out.append({"role": "user", "content": m["content"]})

        flush()
        return out

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4000,
        temperature: float = 0.2,  # accepted and ignored; see below
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": self._messages(messages),
            # Sampling parameters are rejected on current Claude models, so
            # temperature is deliberately not forwarded.
            "thinking": {"type": "adaptive"},
        }
        if tools:
            kwargs["tools"] = _tools_anthropic(tools)

        try:
            resp = await self.client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(_friendly_error(self.provider, exc)) from exc

        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        calls = [
            ToolCall(
                id=b.id,
                name=b.name,
                args=b.input if isinstance(b.input, dict) else json.loads(b.input),
            )
            for b in resp.content
            if b.type == "tool_use"
        ]
        return LLMResponse(text=text, tool_calls=calls)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def _friendly_error(provider: dict, exc: Exception) -> str:
    """Turn a provider exception into something worth showing a user."""
    name = provider["name"]
    text = str(exc)
    lowered = text.lower()

    if "401" in text or "unauthorized" in lowered or "invalid api key" in lowered:
        return f"{name} rejected the API key. Check {provider['key_name']} in your .env."
    if "429" in text or "rate limit" in lowered:
        return (
            f"{name} rate limit reached. Free tiers throttle quickly — wait a "
            f"moment and retry, or switch LLM_PROVIDER in your .env."
        )
    if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
        return (
            f"{name} does not recognise the model '{provider['model']}'. "
            f"Check the model name in your .env — free model ids change often."
        )
    if "timeout" in lowered or "timed out" in lowered:
        return f"{name} timed out. Retry, or switch to a faster provider."
    return f"{name} request failed: {text[:300]}"


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def get_client():
    """Build the client for whichever provider is configured."""
    provider = config.active_provider()

    if not provider["api_key"]:
        raise LLMError(
            f"No API key for provider '{provider['name']}'. Set "
            f"{provider['key_name']} in your .env, or set LLM_PROVIDER to one "
            f"you do have a key for ({', '.join(config.PROVIDERS)})."
        )

    if provider["openai_compatible"]:
        return OpenAICompatibleClient(provider)
    return AnthropicClient(provider)


def describe_provider() -> dict:
    """Reportable provider state for the health endpoint."""
    try:
        provider = config.active_provider()
    except RuntimeError as exc:
        return {"provider": config.LLM_PROVIDER, "error": str(exc)}
    return {
        "provider": provider["name"],
        "model": provider["model"],
        "key_present": bool(provider["api_key"]),
        "key_env_var": provider["key_name"],
    }
