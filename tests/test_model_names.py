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


def test_parse_gemini_3_1_pro_thinking_high():
    spec = parse_model_name("gemini-3.1-pro-thinking-high")
    assert spec.provider == "gemini"
    assert spec.model_id == "gemini-3.1-pro-preview"
    assert spec.thinking == {"thinking_level": "high", "include_thoughts": True}
    assert spec.output_config == {"effort": "high"}
    assert spec.max_tokens == 65_536


def test_parse_gemini_flash_defaults_and_hyphen_version():
    spec = parse_model_name("gemini-3.8-flash")
    assert spec.model_id == "gemini-3.8-flash"
    assert spec.thinking["thinking_level"] == "high"
    assert parse_model_name("gemini-3-1-pro-low").model_id == "gemini-3.1-pro-preview"
    assert parse_model_name("gemini-3-flash-thinking-medium").thinking["thinking_level"] == "medium"
    assert parse_model_name("gemini-9.9-ultra").model_id == "gemini-9.9-ultra"


def test_parse_openrouter_alias_and_raw_slug():
    spec = parse_model_name("glm-5.3-thinking-high")
    assert spec.provider == "openrouter"
    assert spec.model_id == "z-ai/glm-5.3"
    assert spec.output_config == {"effort": "high"}
    assert spec.thinking is None

    # gpt-oss must route to OpenRouter, not the OpenAI Responses API.
    assert parse_model_name("gpt-oss-120b-thinking-medium").provider == "openrouter"
    assert parse_model_name("gpt-oss-120b-thinking-medium").model_id == "openai/gpt-oss-120b"
    assert parse_model_name("gpt-5.6-sol-thinking-high").provider == "openai"

    raw = parse_model_name("openrouter/deepseek/deepseek-v4-pro-thinking-xhigh")
    assert raw.provider == "openrouter"
    assert raw.model_id == "deepseek/deepseek-v4-pro"
    assert raw.output_config == {"effort": "xhigh"}
    assert parse_model_name("openrouter/moonshotai/kimi-k3").output_config == {"effort": "high"}


def test_friendly_names_for_gemini_and_openrouter():
    assert friendly_model_name("gemini-3.1-pro-thinking-high") == "Gemini 3.1 Pro"
    assert friendly_model_name("gemini-3.8-flash-thinking-low") == "Gemini 3.8 Flash Low"
    assert friendly_model_name("glm-5.3-thinking-high") == "GLM 5.3"
    assert friendly_model_name("gpt-oss-120b-thinking-medium") == "GPT OSS 120b Medium"
    assert friendly_model_name("openrouter/deepseek/deepseek-v4-pro-thinking-high") == "Deepseek V4 Pro"


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


def test_parse_newest_frontier_models():
    sol = parse_model_name("gpt-6-sol-thinking-high")
    luna = parse_model_name("gpt-6-luna-thinking-high")
    grok = parse_model_name("grok-4.7-high")
    opus = parse_model_name("claude-opus-5-5-thinking-high")
    assert (sol.provider, sol.model_id, sol.output_config) == (
        "openai",
        "gpt-6-sol",
        {"effort": "high"},
    )
    assert (luna.provider, luna.model_id) == ("openai", "gpt-6-luna")
    assert (grok.provider, grok.model_id, grok.output_config) == (
        "xai",
        "grok-4.7",
        {"effort": "high"},
    )
    assert opus.provider == "anthropic"
    assert opus.model_id == "claude-opus-5-5"
    assert opus.thinking == {"type": "adaptive", "display": "summarized"}
    assert opus.output_config == {"effort": "high"}
    assert opus.max_tokens == 128_000
    assert friendly_model_name("claude-opus-5-5-thinking-high") == "Opus 5.5"
    assert friendly_model_name("gpt-6-sol-thinking-high") == "GPT 6 Sol"
    assert friendly_model_name("gpt-6-luna-thinking-high") == "GPT 6 Luna"
    assert friendly_model_name("grok-4.7-high") == "Grok 4.7"


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
