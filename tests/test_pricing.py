from harness.pricing import estimate_cost, infer_cache_ttl


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
