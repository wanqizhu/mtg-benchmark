from harness.pricing import (
    estimate_cost,
    estimate_rollout_cost,
    infer_cache_ttl,
    reclassify_1h_cache_write_as_read,
    reclassify_cache_as_input,
    reclassify_tools_result_cache_as_input,
)


def test_sonnet_tools_run_cost():
    usage = {
        "input_tokens": 5,
        "output_tokens": 47811,
        "cache_creation_input_tokens": 53892,
        "cache_read_input_tokens": 98964,
    }
    cost = estimate_cost(usage, model_id="claude-sonnet-4-6", cache_ttl="5m")
    assert cost["usd"] > 0
    assert cost["breakdown_usd"]["output"] > cost["breakdown_usd"]["input"]
    assert cost["breakdown_usd"]["cache_read"] > 0
    assert cost["breakdown_usd"]["cache_creation"] > 0


def test_inline_cache_write_uses_1h_rate():
    usage = {
        "input_tokens": 2,
        "output_tokens": 49746,
        "cache_creation_input_tokens": 232405,
        "cache_read_input_tokens": 0,
    }
    cost_1h = estimate_cost(usage, model_id="claude-sonnet-4-6", cache_ttl="1h")
    cost_5m = estimate_cost(usage, model_id="claude-sonnet-4-6", cache_ttl="5m")
    assert cost_1h["breakdown_usd"]["cache_creation"] > cost_5m["breakdown_usd"]["cache_creation"]


def test_sonnet_5_pricing():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    cost = estimate_cost(usage, model_id="claude-sonnet-5", cache_ttl="1h")
    assert cost["usd"] == 12.0
    assert cost["breakdown_usd"]["input"] == 2.0
    assert cost["breakdown_usd"]["output"] == 10.0


def test_empty_usage_is_zero():
    assert estimate_cost({}, model_id="claude-sonnet-4-6")["usd"] == 0.0


def test_infer_cache_ttl_from_rules_mode():
    assert infer_cache_ttl({"rules_mode": "inline"}) == "1h"
    assert infer_cache_ttl({"rules_mode": "tools"}) == "5m"
    assert infer_cache_ttl({}, result_path="results/inline-rules/foo.json") == "1h"


def test_reclassify_cache_as_input():
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_creation_input_tokens": 1000,
        "cache_read_input_tokens": 200,
        "cache_creation": {
            "ephemeral_5m_input_tokens": 1000,
            "ephemeral_1h_input_tokens": 0,
        },
    }
    moved = reclassify_cache_as_input(usage)
    assert moved == 1200
    assert usage["input_tokens"] == 1300
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 0
    assert usage["cache_creation"]["ephemeral_5m_input_tokens"] == 0
    assert reclassify_cache_as_input(usage) == 0


def test_reclassify_tools_result_recomputes_cost():
    payload = {
        "model_id": "claude-sonnet-5",
        "transcript": {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 0,
                "cache_creation_input_tokens": 900,
                "cache_read_input_tokens": 0,
            },
            "turns": [
                {
                    "role": "assistant",
                    "usage": {
                        "input_tokens": 100,
                        "cache_creation_input_tokens": 900,
                        "cache_read_input_tokens": 0,
                    },
                }
            ],
        },
    }
    moved = reclassify_tools_result_cache_as_input(payload)
    assert moved == 900
    assert payload["transcript"]["usage"]["input_tokens"] == 1000
    assert payload["transcript"]["turns"][0]["usage"]["input_tokens"] == 1000
    assert payload["cost"]["breakdown_usd"]["input"] == 0.002
    assert payload["cost"]["breakdown_usd"]["cache_creation"] == 0.0


def test_reclassify_1h_cache_write_as_read():
    usage = {
        "input_tokens": 882,
        "output_tokens": 36180,
        "cache_creation_input_tokens": 311809,
        "cache_read_input_tokens": 623618,
        "cache_creation": {
            "ephemeral_1h_input_tokens": 311809,
            "ephemeral_5m_input_tokens": 0,
        },
    }
    moved = reclassify_1h_cache_write_as_read(usage)
    assert moved == 311809
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 935427
    assert usage["cache_creation"]["ephemeral_1h_input_tokens"] == 0
    assert reclassify_1h_cache_write_as_read(usage) == 0


def test_grok_standard_pricing():
    usage = {
        "input_tokens": 100_000,
        "output_tokens": 100_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "prompt_tokens": 100_000,
    }
    cost = estimate_cost(usage, model_id="grok-4.6", cache_ttl="auto")
    assert cost["usd"] == 0.8
    assert cost["breakdown_usd"]["input"] == 0.2
    assert cost["breakdown_usd"]["output"] == 0.6
    assert cost["breakdown_usd"]["cache_read"] == 0.0
    assert cost["long_context"] is False
    assert cost["pricing_source"].endswith("/pricing")


def test_grok_long_context_doubles_rates():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "prompt_tokens": 1_000_000,
    }
    cost = estimate_cost(usage, model_id="grok-4.6", cache_ttl="auto")
    assert cost["long_context"] is True
    assert cost["usd"] == 16.0
    assert cost["breakdown_usd"]["input"] == 4.0
    assert cost["breakdown_usd"]["output"] == 12.0


def test_grok_cached_input_and_short_context():
    usage = {
        "input_tokens": 50_000,
        "output_tokens": 10_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 40_000,
        "prompt_tokens": 90_000,
    }
    cost = estimate_cost(usage, model_id="grok-4.6", cache_ttl="auto")
    assert cost["long_context"] is False
    assert cost["breakdown_usd"]["input"] == 0.1
    assert cost["breakdown_usd"]["cache_read"] == 0.02
    assert cost["breakdown_usd"]["output"] == 0.06
    assert cost["usd"] == 0.18


def test_grok_rollout_cost_applies_long_context_per_turn():
    turns = [
        {
            "role": "assistant",
            "usage": {
                "input_tokens": 10_000,
                "output_tokens": 1_000,
                "cache_read_input_tokens": 0,
                "prompt_tokens": 10_000,
            },
        },
        {
            "role": "assistant",
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 1_000,
                "cache_read_input_tokens": 199_000,
                "prompt_tokens": 200_000,
            },
        },
    ]
    cost = estimate_rollout_cost(
        usage={},
        turns=turns,
        model_id="grok-4.6",
        cache_ttl="auto",
    )
    # Turn 1: 10k * $2 + 1k * $6 = 0.026
    # Turn 2 (2x): 1k * $4 + 199k * $1 + 1k * $12 = 0.004 + 0.199 + 0.012 = 0.215
    assert cost["long_context"] is True
    assert cost["usd"] == 0.241
    assert cost["breakdown_usd"]["cache_read"] == 0.199
