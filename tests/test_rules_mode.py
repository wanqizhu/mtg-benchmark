from benchmarks.mtg.bench import MTGBenchmark
from harness.model_names import parse_model_name
from harness.providers.anthropic import (
    PREWARM_MAX_TOKENS,
    _apply_cache_control,
    _cache_control,
    _prewarm_request,
    _system_blocks,
)


def test_inline_samples_have_no_tools():
    bench = MTGBenchmark(rules_mode="inline")
    sample = bench.samples()[0]
    assert sample.tools == []
    assert sample.reference["rules_mode"] == "inline"
    assert "<comprehensive_rules>" in sample.system
    assert "grep(" not in sample.system


def test_tools_samples_have_tools():
    bench = MTGBenchmark(rules_mode="tools")
    sample = bench.samples()[0]
    assert len(sample.tools) == 2
    assert sample.reference["rules_mode"] == "tools"
    assert "grep(" in sample.system


def test_inline_rejects_haiku():
    bench = MTGBenchmark(rules_mode="inline")
    spec = parse_model_name("claude-haiku-4-5-thinking")
    try:
        bench.validate_model(spec)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_inline_allows_sonnet_46():
    bench = MTGBenchmark(rules_mode="inline")
    spec = parse_model_name("claude-sonnet-4-6-thinking-high")
    bench.validate_model(spec)


def test_inline_allows_sonnet_5():
    bench = MTGBenchmark(rules_mode="inline")
    spec = parse_model_name("claude-sonnet-5-thinking-high")
    bench.validate_model(spec)


def test_prompt_cache_is_one_hour():
    assert _cache_control() == {"type": "ephemeral", "ttl": "1h"}


def test_prewarm_request_matches_rollout_cache_key():
    high = parse_model_name("claude-sonnet-5-thinking-high")
    low = parse_model_name("claude-sonnet-5-thinking-low")
    high_req = _prewarm_request(spec=high, system="rules")
    low_req = _prewarm_request(spec=low, system="rules")
    assert high_req["max_tokens"] == PREWARM_MAX_TOKENS
    assert high_req["max_tokens"] < high.max_tokens
    assert high_req["thinking"] == high.thinking
    assert high_req["output_config"] == {"effort": "high"}
    assert low_req["output_config"] == {"effort": "low"}
    assert high_req["thinking"] == low_req["thinking"]
    assert high_req["output_config"] != low_req["output_config"]
    assert high_req["system"][0]["cache_control"]["ttl"] == "1h"


def test_prewarm_request_omits_thinking_when_disabled():
    spec = parse_model_name("claude-sonnet-5")
    request = _prewarm_request(spec=spec, system="rules")
    assert "thinking" not in request
    assert "output_config" not in request


def test_inline_one_shot_does_not_cache_user_prompt():
    messages = [{"role": "user", "content": [{"type": "text", "text": "puzzle"}]}]
    cached = _apply_cache_control(messages)
    assert "cache_control" not in cached[0]["content"][0]


def test_tools_does_not_cache_messages():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "puzzle"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "rules"}],
        },
    ]
    cached = _apply_cache_control(messages)
    assert "cache_control" not in cached[0]["content"][0]
    assert "cache_control" not in cached[2]["content"][0]


def test_tools_does_not_cache_system_prompt():
    assert "cache_control" not in _system_blocks("tools system", cache=False)[0]
    assert _system_blocks("inline rules", cache=True)[0]["cache_control"]["ttl"] == "1h"


def test_default_judge_model_is_sonnet():
    from harness.config import DEFAULT_JUDGE_MODEL

    assert DEFAULT_JUDGE_MODEL.startswith("claude-sonnet-5")
