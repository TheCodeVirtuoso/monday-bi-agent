"""Tests for the provider-agnostic LLM layer.

The translation between the neutral conversation format and each provider's
wire format is the riskiest code in the project: it is exercised on every
turn, and a mistake shows up as a confusing model failure rather than a
stack trace. It is tested here without any network access.
"""

import json

import pytest

import config
import llm
import tools as T


# --------------------------------------------------------------------------
# Tool schema translation
# --------------------------------------------------------------------------


ALL_TOOLS = T.DEALS_TOOLS + T.WORK_ORDER_TOOLS + T.ORCHESTRATOR_TOOLS


def test_every_tool_is_declared_in_the_neutral_shape():
    for tool in ALL_TOOLS:
        assert set(tool) == {"name", "description", "parameters"}, tool["name"]
        assert tool["parameters"]["type"] == "object"
        assert tool["description"].strip()


def test_openai_translation_wraps_in_function_envelope():
    out = llm._tools_openai(T.DEALS_TOOLS)
    assert all(t["type"] == "function" for t in out)
    assert {t["function"]["name"] for t in out} == {
        t["name"] for t in T.DEALS_TOOLS
    }
    assert out[0]["function"]["parameters"] == T.DEALS_TOOLS[0]["parameters"]


def test_anthropic_translation_renames_parameters_to_input_schema():
    out = llm._tools_anthropic(T.DEALS_TOOLS)
    assert all("input_schema" in t and "parameters" not in t for t in out)
    assert out[0]["input_schema"] == T.DEALS_TOOLS[0]["parameters"]


def test_required_fields_survive_translation():
    snapshot = next(t for t in T.DEALS_TOOLS if t["name"] == "deals_snapshot")
    assert snapshot["parameters"]["required"] == ["kind"]
    assert (
        llm._tools_openai([snapshot])[0]["function"]["parameters"]["required"]
        == ["kind"]
    )
    assert llm._tools_anthropic([snapshot])[0]["input_schema"]["required"] == ["kind"]


# --------------------------------------------------------------------------
# Message translation
# --------------------------------------------------------------------------


@pytest.fixture
def conversation():
    call_a = llm.ToolCall(id="c1", name="analyze_deals", args={"question": "pipeline?"})
    call_b = llm.ToolCall(id="c2", name="analyze_work_orders", args={"question": "cash?"})
    return [
        llm.user("How are we doing?"),
        llm.assistant(llm.LLMResponse(text="Checking.", tool_calls=[call_a, call_b])),
        llm.tool_result(call_a, '{"findings": "deals ok"}'),
        llm.tool_result(call_b, '{"findings": "cash ok"}'),
    ], (call_a, call_b)


def test_openai_messages_keep_system_first_and_tools_separate(conversation):
    messages, _ = conversation
    out = llm.OpenAICompatibleClient._messages("SYS", messages)

    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1]["role"] == "user"

    tool_calls = out[2]["tool_calls"]
    assert [c["id"] for c in tool_calls] == ["c1", "c2"]
    # OpenAI wants arguments as a JSON *string*, not an object.
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"question": "pipeline?"}

    assert [m["role"] for m in out[3:]] == ["tool", "tool"]
    assert out[3]["tool_call_id"] == "c1"


def test_anthropic_coalesces_tool_results_into_one_user_message(conversation):
    """Splitting results across messages trains the model out of parallel calls."""
    messages, _ = conversation
    out = llm.AnthropicClient._messages(messages)

    assert [m["role"] for m in out] == ["user", "assistant", "user"]

    blocks = out[1]["content"]
    assert blocks[0]["type"] == "text"
    assert [b["id"] for b in blocks if b["type"] == "tool_use"] == ["c1", "c2"]
    # Anthropic wants the arguments as an object, not a string.
    assert blocks[1]["input"] == {"question": "pipeline?"}

    results = out[2]["content"]
    assert len(results) == 2
    assert all(b["type"] == "tool_result" for b in results)
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]


def test_anthropic_assistant_turn_with_no_text_still_has_content():
    """An empty content list is rejected by the API."""
    call = llm.ToolCall(id="x", name="analyze_deals", args={})
    out = llm.AnthropicClient._messages(
        [llm.assistant(llm.LLMResponse(text="", tool_calls=[call]))]
    )
    assert out[0]["content"]


def test_round_trip_preserves_call_ids_across_both_providers(conversation):
    messages, (call_a, call_b) = conversation
    openai_out = llm.OpenAICompatibleClient._messages("SYS", messages)
    anthropic_out = llm.AnthropicClient._messages(messages)

    openai_ids = [m["tool_call_id"] for m in openai_out if m["role"] == "tool"]
    anthropic_ids = [b["tool_use_id"] for b in anthropic_out[2]["content"]]
    assert openai_ids == anthropic_ids == [call_a.id, call_b.id]


# --------------------------------------------------------------------------
# Provider selection and errors
# --------------------------------------------------------------------------


def test_auto_provider_prefers_a_key_that_exists(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-or-test")
    assert config._auto_provider() == "openrouter"


def test_unknown_provider_is_rejected_by_name(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "gpt5-turbo-max")
    with pytest.raises(RuntimeError) as exc:
        config.active_provider()
    assert "gpt5-turbo-max" in str(exc.value)


def test_missing_key_error_names_the_env_var_to_set(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
    monkeypatch.setitem(config.PROVIDERS["groq"], "api_key", "")
    with pytest.raises(llm.LLMError) as exc:
        llm.get_client()
    assert "GROQ_API_KEY" in str(exc.value)


@pytest.mark.parametrize(
    "raw,expected_phrase",
    [
        ("Error code: 401 unauthorized", "rejected the API key"),
        ("429 Too Many Requests: rate limit exceeded", "rate limit"),
        ("The model `foo` does not exist", "does not recognise the model"),
        ("Request timed out", "timed out"),
    ],
)
def test_provider_errors_become_actionable_messages(raw, expected_phrase):
    provider = {"name": "groq", "model": "llama-3.3-70b-versatile", "key_name": "GROQ_API_KEY"}
    assert expected_phrase in llm._friendly_error(provider, Exception(raw))


def test_response_reports_whether_tools_were_requested():
    assert not llm.LLMResponse(text="hi").wants_tools
    assert llm.LLMResponse(
        tool_calls=[llm.ToolCall(id="1", name="analyze_deals", args={})]
    ).wants_tools
