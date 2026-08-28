from harness.pricing import (
    estimate_cost,
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
