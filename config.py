"""Central configuration. Everything reads env once, here."""

import os

from dotenv import load_dotenv

load_dotenv()


def env(name: str, default: str = "") -> str:
    """Read an env var, stripping surrounding whitespace.

    Hosting dashboards use multi-line textareas for secrets, so a pasted
    token routinely arrives with a trailing newline. That is invisible in the
    UI and fatal in an HTTP header - httpx rejects the value outright, and the
    failure surfaces as "Illegal header value" rather than as anything that
    points at the real cause. Strip once, here.
    """
    return (os.getenv(name) or default).strip()


# --------------------------------------------------------------------------
# LLM provider
# --------------------------------------------------------------------------
# Three interchangeable backends. Groq and OpenRouter are OpenAI-compatible
# and reached through the same client; Anthropic uses its own SDK. Which one
# is active is a config choice, not a code change - see llm.py.

GROQ_API_KEY = env("GROQ_API_KEY")
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY")

# Defaults chosen for tool-calling reliability, which is what this app leans
# on. Everything else (math, dates, normalisation) is deterministic Python,
# so raw model intelligence matters far less here than schema adherence.
#
# Verified against this account's live Groq catalogue. Note Groq has RETIRED
# the llama-3.x ids that most tutorials still reference - they now 404. Models
# confirmed to emit well-formed tool calls here:
#   openai/gpt-oss-120b   (default; strongest)
#   openai/gpt-oss-20b    (smaller, faster)
#   qwen/qwen3.8-27b      (works)
#   qwen/qwen3.6-27b      (does NOT reliably call tools - avoid)
GROQ_MODEL = env("GROQ_MODEL", "openai/gpt-oss-120b")
OPENROUTER_MODEL = env("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
ANTHROPIC_MODEL = env("ANTHROPIC_MODEL", "claude-opus-5")

# Groq meters 200,000 tokens per day PER MODEL, so an exhausted model has
# healthy siblings. Falling back to one recovers instantly, where waiting for
# the daily window would not.
GROQ_FALLBACK_MODELS = [
    m.strip() for m in env(
        "GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,qwen/qwen3.8-27b"
    ).split(",") if m.strip()
]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _auto_provider() -> str:
    """Pick a provider from whichever credential is actually present."""
    if GROQ_API_KEY:
        return "groq"
    if OPENROUTER_API_KEY:
        return "openrouter"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "groq"  # so the error message names a concrete key to set


LLM_PROVIDER = (env("LLM_PROVIDER") or _auto_provider()).lower()

PROVIDERS = {
    "groq": {
        "api_key": GROQ_API_KEY,
        "model": GROQ_MODEL,
        "fallback_models": GROQ_FALLBACK_MODELS,
        "base_url": GROQ_BASE_URL,
        "key_name": "GROQ_API_KEY",
        "openai_compatible": True,
    },
    "openrouter": {
        "api_key": OPENROUTER_API_KEY,
        "model": OPENROUTER_MODEL,
        "base_url": OPENROUTER_BASE_URL,
        "key_name": "OPENROUTER_API_KEY",
        "openai_compatible": True,
    },
    "anthropic": {
        "api_key": ANTHROPIC_API_KEY,
        "model": ANTHROPIC_MODEL,
        "base_url": None,
        "key_name": "ANTHROPIC_API_KEY",
        "openai_compatible": False,
    },
}


def active_provider() -> dict:
    if LLM_PROVIDER not in PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
            f"Choose one of: {', '.join(PROVIDERS)}."
        )
    return {"name": LLM_PROVIDER, **PROVIDERS[LLM_PROVIDER]}


def llm_configured() -> bool:
    return bool(PROVIDERS.get(LLM_PROVIDER, {}).get("api_key"))


# --------------------------------------------------------------------------
# monday.com
# --------------------------------------------------------------------------

MONDAY_API_URL = "https://api.monday.com/v2"
MONDAY_API_TOKEN = env("MONDAY_API_TOKEN")
WORK_ORDERS_BOARD_ID = env("MONDAY_WORK_ORDERS_BOARD_ID")
DEALS_BOARD_ID = env("MONDAY_DEALS_BOARD_ID")

# File mode is on if explicitly requested OR if we simply have no monday
# credentials. This keeps the app fully runnable before the boards exist.
USE_MOCK_DATA = (
    env("USE_MOCK_DATA").lower() in {"1", "true", "yes"}
    or not MONDAY_API_TOKEN
    or not WORK_ORDERS_BOARD_ID
    or not DEALS_BOARD_ID
)

MONDAY_TIMEOUT_SECONDS = float(env("MONDAY_TIMEOUT_SECONDS", "20"))
