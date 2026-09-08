from harness.model_names import friendly_model_name, parse_model_name


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


def test_parse_fable_5_1_thinking_high():
    spec = parse_model_name("claude-fable-5-1-thinking-high")
    assert spec.model_id == "claude-fable-5-1"
    assert spec.thinking == {"type": "adaptive", "display": "summarized"}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_parse_fable_5_1_dotted_version():
    spec = parse_model_name("claude-fable-5.1-thinking-high")
    assert spec.model_id == "claude-fable-5-1"
    assert spec.thinking == {"type": "adaptive", "display": "summarized"}
    assert spec.max_tokens == 128_000


def test_parse_fable_5_still_maps_without_point_release():
    spec = parse_model_name("claude-fable-5-thinking-high")
    assert spec.model_id == "claude-fable-5"
    assert spec.thinking == {"type": "adaptive", "display": "summarized"}
    assert spec.output_config == {"effort": "high"}


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


def test_parse_gpt_5_6_sol_thinking_high():
    spec = parse_model_name("gpt-5.6-sol-thinking-high")
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-5.6-sol"
    assert spec.thinking == {"summary": "auto", "context": "all_turns"}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_parse_gpt_defaults_to_medium_without_summary():
    spec = parse_model_name("gpt-5.6-sol")
    assert spec.provider == "openai"
    assert spec.thinking is None
    assert spec.output_config == {"effort": "medium"}


def test_parse_gpt_5_6_terra_thinking_high():
    spec = parse_model_name("gpt-5.6-terra-thinking-high")
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-5.6-terra"
    assert spec.thinking == {"summary": "auto", "context": "all_turns"}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_parse_gpt_5_6_luna_thinking_high():
    spec = parse_model_name("gpt-5.6-luna-thinking-high")
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-5.6-luna"
    assert spec.thinking == {"summary": "auto", "context": "all_turns"}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_parse_gpt_6_astra_thinking_high():
    spec = parse_model_name("gpt-6-astra-thinking-high")
    assert spec.provider == "openai"
    assert spec.model_id == "gpt-6-astra"
    assert spec.thinking == {"summary": "auto", "context": "all_turns"}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 128_000


def test_friendly_model_name_drops_thinking_and_high():
    assert friendly_model_name("claude-fable-5-1-thinking-high") == "Fable 5.1"
    assert friendly_model_name("claude-opus-5-thinking-high") == "Opus 5"
    assert friendly_model_name("claude-sonnet-5-thinking-high") == "Sonnet 5"
    assert friendly_model_name("claude-haiku-4-5-thinking") == "Haiku 4.5"
    assert friendly_model_name("gpt-5.6-sol-thinking-high") == "GPT 5.6 Sol"
    assert friendly_model_name("gpt-5.6-terra-thinking-high") == "GPT 5.6 Terra"
    assert friendly_model_name("gpt-5.6-luna-thinking-high") == "GPT 5.6 Luna"
    assert friendly_model_name("gpt-6-astra-thinking-high") == "GPT 6 Astra"
    assert friendly_model_name("grok-4.6-high") == "Grok 4.6"
    assert friendly_model_name("grok-4.5-high") == "Grok 4.5"


def test_friendly_model_name_keeps_non_high_effort():
    assert friendly_model_name("claude-sonnet-5-thinking-low") == "Sonnet 5 Low"
    assert friendly_model_name("claude-opus-4-8-thinking-xhigh") == "Opus 4.8 Xhigh"
    assert friendly_model_name("claude-fable-5-1-thinking-high @ grep rules") == "Fable 5.1 @ grep rules"
