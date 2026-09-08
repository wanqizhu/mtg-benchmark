from benchmarks.mtg.bench import MTGBenchmark
from harness.model_names import parse_model_name
from harness.providers.anthropic import (
    CONVERSATION_CACHE_MODELS,
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


def test_inline_allows_fable_51():
    bench = MTGBenchmark(rules_mode="inline")
    spec = parse_model_name("claude-fable-5-1-thinking-high")
    bench.validate_model(spec)


def test_inline_allows_grok_46():
    bench = MTGBenchmark(rules_mode="inline")
    spec = parse_model_name("grok-4.6-high")
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


def _conversation() -> list[dict]:
    return [
        {"role": "user", "content": [{"type": "text", "text": "puzzle"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "thinking"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "rules"}],
        },
    ]


def test_inline_one_shot_does_not_cache_user_prompt():
    messages = [{"role": "user", "content": [{"type": "text", "text": "puzzle"}]}]
    cached = _apply_cache_control(messages, cache=False)
    assert "cache_control" not in cached[0]["content"][0]


def test_uncached_models_do_not_cache_messages():
    cached = _apply_cache_control(_conversation(), cache=False)
    assert "cache_control" not in cached[0]["content"][0]
    assert "cache_control" not in cached[2]["content"][0]


def test_rolling_breakpoint_marks_only_the_latest_turn():
    cached = _apply_cache_control(_conversation(), cache=True)
    assert cached[2]["content"][-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in cached[0]["content"][0]
    assert "cache_control" not in cached[1]["content"][0]


def test_rolling_breakpoint_moves_and_leaves_no_stale_markers():
    """Turn N's breakpoint must be dropped when turn N+1 adds its own."""
    messages = _apply_cache_control(_conversation(), cache=True)
    messages.append(
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "2", "content": "more"}]}
    )
    cached = _apply_cache_control(messages, cache=True)
    marked = [i for i, m in enumerate(cached) if "cache_control" in m["content"][-1]]
    assert marked == [3]


def test_apply_cache_control_does_not_mutate_caller_messages():
    messages = _conversation()
    _apply_cache_control(messages, cache=True)
    assert "cache_control" not in messages[-1]["content"][-1]


def test_system_prompt_caching_follows_the_cache_flag():
    assert "cache_control" not in _system_blocks("tools system", cache=False)[0]
    assert _system_blocks("inline rules", cache=True)[0]["cache_control"]["ttl"] == "1h"


def test_conversation_cache_is_enabled_only_for_sonnet_5_and_opus_5():
    cached = {parse_model_name(n).model_id for n in ("claude-sonnet-5", "claude-opus-5")}
    assert CONVERSATION_CACHE_MODELS == cached
    # Haiku's 4096-token minimum exceeds the tools-mode prefix, and short
    # rollouts lose money on the write premium.
    assert parse_model_name("claude-haiku-4-5-thinking").model_id not in CONVERSATION_CACHE_MODELS
    assert parse_model_name("claude-fable-5-1-thinking-high").model_id not in CONVERSATION_CACHE_MODELS


def test_default_judge_model_is_sol():
    from harness.config import DEFAULT_JUDGE_MODEL

    assert DEFAULT_JUDGE_MODEL == "gpt-5.6-sol-thinking-high"
