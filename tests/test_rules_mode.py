from benchmarks.mtg.bench import MTGBenchmark
from harness.model_names import parse_model_name
from harness.providers.anthropic import (
    PREWARM_MAX_TOKENS,
    _apply_cache_control,
    _cache_control,
    _prewarm_request,
)
from harness.runner import Runner


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


def test_inline_runner_auto_uses_one_hour_cache():
    bench = MTGBenchmark(rules_mode="inline")
    runner = Runner(bench, run_id="test", rules_mode="inline")
    assert runner.cache_ttl == "1h"


def test_tools_runner_auto_uses_five_minute_cache():
    bench = MTGBenchmark(rules_mode="tools")
    runner = Runner(bench, run_id="test", rules_mode="tools")
    assert runner.cache_ttl == "5m"


def test_one_hour_cache_control():
    assert _cache_control("1h") == {"type": "ephemeral", "ttl": "1h"}
    assert _cache_control("5m") == {"type": "ephemeral"}


def test_prewarm_request_is_cheap():
    spec = parse_model_name("claude-sonnet-5-thinking-high")
    request = _prewarm_request(spec=spec, system="rules", cache_ttl="1h")
    assert request["max_tokens"] == PREWARM_MAX_TOKENS
    assert request["max_tokens"] < spec.max_tokens
    assert "thinking" not in request
    assert "output_config" not in request
    assert request["system"][0]["cache_control"]["ttl"] == "1h"


def test_inline_one_shot_does_not_cache_user_prompt():
    messages = [{"role": "user", "content": [{"type": "text", "text": "puzzle"}]}]
    cached = _apply_cache_control(messages, cache_ttl="1h", cache_first_user=False)
    assert "cache_control" not in cached[0]["content"][0]


def test_default_judge_model_is_sonnet():
    from harness.config import DEFAULT_JUDGE_MODEL

    assert DEFAULT_JUDGE_MODEL.startswith("claude-sonnet-5")
