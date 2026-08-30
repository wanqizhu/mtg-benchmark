from harness.model_names import parse_model_name


def test_parse_basic():
    spec = parse_model_name("claude-sonnet-4-6")
    assert spec.model_id == "claude-sonnet-4-6"
    assert spec.thinking is None
    assert spec.output_config is None
    assert spec.max_tokens == 128_000


def test_parse_thinking_and_effort():
    spec = parse_model_name("claude-opus-4-8-thinking-xhigh")
    assert spec.model_id == "claude-opus-4-8"
    assert spec.thinking == {"type": "adaptive", "display": "summarized"}
    assert spec.output_config == {"effort": "xhigh"}
    assert spec.max_tokens == 128_000


def test_parse_sonnet_5_thinking_high():
    spec = parse_model_name("claude-sonnet-5-thinking-high")
    assert spec.model_id == "claude-sonnet-5"
    assert spec.thinking == {"type": "adaptive", "display": "summarized"}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_parse_haiku_maps_to_dated_id():
    spec = parse_model_name("claude-haiku-4-5-thinking")
    assert spec.model_id == "claude-haiku-4-5-20251001"
    assert spec.thinking == {
        "type": "enabled",
        "budget_tokens": 10_000,
        "display": "summarized",
    }
    assert spec.max_tokens == 64_000


def test_parse_effort_order_independent():
    spec = parse_model_name("claude-sonnet-4-6-high-thinking")
    assert spec.output_config == {"effort": "high"}
    assert spec.thinking == {"type": "adaptive", "display": "summarized"}
    assert spec.max_tokens == 128_000


def test_older_sonnet_and_opus_use_64k_output_limit():
    assert parse_model_name("claude-sonnet-4-5-thinking").max_tokens == 64_000
    assert parse_model_name("claude-opus-4-5-thinking-high").max_tokens == 64_000


def test_parse_grok_4_6_high():
    spec = parse_model_name("grok-4.6-high")
    assert spec.provider == "xai"
    assert spec.model_id == "grok-4.6"
    assert spec.thinking is None
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_parse_grok_hyphen_version_and_default_effort():
    spec = parse_model_name("grok-4-6")
    assert spec.model_id == "grok-4.6"
    assert spec.output_config == {"effort": "high"}


def test_parse_grok_thinking_suffix_ignored():
    spec = parse_model_name("grok-4.6-thinking-xhigh")
    assert spec.model_id == "grok-4.6"
    assert spec.output_config == {"effort": "xhigh"}


def test_parse_grok_rejects_max_effort():
    try:
        parse_model_name("grok-4.6-max")
        raised = False
    except ValueError:
        raised = True
    assert raised
