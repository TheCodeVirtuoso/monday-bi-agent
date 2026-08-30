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

import asyncio
import json
import re
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
        self.fallback_models = provider.get("fallback_models") or []
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

    async def _create_with_backoff(self, kwargs: dict):
        """Call the provider, surviving a rate-limited model.

        Groq's free tier meters two separate budgets: 8,000 tokens per minute
        and 200,000 per day, and the daily one is per MODEL. So when the
        primary model is exhausted a sibling usually still has headroom —
        switching to it recovers instantly, where waiting would not.

        A short wait is only worth it for the per-minute limit, which clears
        in seconds. The daily limit reports ``x-should-retry: false`` and a
        multi-minute delay; there is no point sleeping on that, so we fall
        back or fail fast with a message that says which budget ran out.
        """
        models = [kwargs["model"]] + [
            m for m in self.fallback_models if m != kwargs["model"]
        ]
        last: Exception | None = None

        for model in models:
            attempt_kwargs = {**kwargs, "model": model}
            for attempt in range(2):
                try:
                    return await self.client.chat.completions.create(**attempt_kwargs)
                except Exception as exc:
                    text = str(exc)
                    if not ("429" in text or "rate limit" in text.lower()):
                        raise
                    last = exc

                    per_day = "per day" in text.lower() or "tpd" in text.lower()
                    should_retry = str(
                        _header(exc, "x-should-retry", "true")
                    ).lower() != "false"

                    if per_day or not should_retry or attempt == 1:
                        break  # try the next model instead of waiting

                    await asyncio.sleep(min(_reset_seconds(exc), 30))

        raise last or LLMError("rate limited on every configured model")

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

        # Reasoning models on Groq (the gpt-oss family, and Qwen3) return their
        # thinking in a separate `reasoning` field and leave `content` null.
        # Asking for it to be hidden puts the actual answer back in `content`,
        # which is where every OpenAI-compatible client looks. Without this the
        # final turn of a conversation can come back completely empty.
        if self.provider["name"] == "groq":
            kwargs["extra_body"] = {"reasoning_format": "hidden"}

        try:
            resp = await self._create_with_backoff(kwargs)
        except LLMError:
            raise
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

        # Belt and braces: if a provider still withholds `content` and puts the
        # text in a reasoning field, use that rather than returning nothing.
        text = (msg.content or "").strip()
        if not text and not calls:
            extra = getattr(msg, "model_extra", None) or {}
            for key in ("reasoning", "reasoning_content"):
                fallback = getattr(msg, key, None) or extra.get(key)
                if fallback:
                    text = str(fallback).strip()
                    break

        return LLMResponse(text=text, tool_calls=calls)


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


def _header(exc: Exception, name: str, default: str = "") -> str:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    return headers.get(name, default)


def _reset_seconds(exc: Exception, default: float = 12.0) -> float:
    """How long the provider says to wait, from its rate-limit headers.

    Groq reports ``x-ratelimit-reset-tokens: 6.112s`` (also ``1m26.4s``).
    Falls back to a fixed pause when no usable header is present.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for name in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(name)
        if not raw:
            continue
        m = re.fullmatch(r"(?:(\d+)m)?([\d.]+)s?", str(raw).strip())
        if m:
            minutes = float(m.group(1) or 0)
            return minutes * 60 + float(m.group(2)) + 1
    return default


def _friendly_error(provider: dict, exc: Exception) -> str:
    """Turn a provider exception into something worth showing a user."""
    name = provider["name"]
    text = str(exc)
    lowered = text.lower()

    if "401" in text or "unauthorized" in lowered or "invalid api key" in lowered:
        return f"{name} rejected the API key. Check {provider['key_name']} in your .env."
    if "429" in text or "rate limit" in lowered:
        # Groq's own message names the budget (per-minute vs per-day), the
        # numbers, and how long to wait. That is far more actionable than
        # anything paraphrased, so quote it.
        detail = ""
        match = re.search(r"Rate limit reached[^\"']*", text)
        if match:
            detail = " " + match.group(0).split("Need more tokens?")[0].strip()
        if "per day" in lowered or "tpd" in lowered:
            return (
                f"{name}'s daily token budget for this model is used up."
                f"{detail} Each model has its own daily allowance, so set "
                f"GROQ_MODEL to another one (or wait for the window to roll)."
            )
        return (
            f"{name} rate limit reached.{detail} Free tiers throttle quickly — "
            f"retry shortly, or switch LLM_PROVIDER."
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
