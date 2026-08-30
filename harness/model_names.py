from __future__ import annotations

import re
from dataclasses import dataclass

FAMILIES = frozenset({"haiku", "sonnet", "opus", "fable"})
EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
GROK_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh"})

# (family, version) -> Anthropic API model id
API_MODEL_IDS: dict[tuple[str, str], str] = {
    ("haiku", "4-5"): "claude-haiku-4-5-20251001",
    ("sonnet", "5"): "claude-sonnet-5",
    ("sonnet", "4-6"): "claude-sonnet-4-6",
    ("sonnet", "4-5"): "claude-sonnet-4-5-20250929",
    ("opus", "4-8"): "claude-opus-4-8",
    ("opus", "4-7"): "claude-opus-4-7",
    ("opus", "4-6"): "claude-opus-4-6",
    ("opus", "4-5"): "claude-opus-4-5-20251101",
    ("fable", "5"): "claude-fable-5",
}

# Models that use adaptive thinking when -thinking is set.
ADAPTIVE_THINKING_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
    }
)

# Models that support output_config.effort.
EFFORT_MODELS = frozenset(
    {
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-4-5-20251101",
    }
)

DEFAULT_THINKING_BUDGET = 10_000

# Synchronous Messages API max output tokens per model.
# https://platform.claude.com/docs/en/about-claude/models/overview
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {
    "claude-fable-5": 128_000,
    "claude-sonnet-5": 128_000,
    "claude-sonnet-4-6": 128_000,
    "claude-opus-4-8": 128_000,
    "claude-opus-4-7": 128_000,
    "claude-opus-4-6": 128_000,
    "claude-sonnet-4-5-20250929": 64_000,
    "claude-opus-4-5-20251101": 64_000,
    "claude-haiku-4-5-20251001": 64_000,
}
FALLBACK_MAX_OUTPUT_TOKENS = 64_000

# (version) -> xAI API model id. Friendly names use grok-{version}[-{effort}].
GROK_API_MODEL_IDS: dict[str, str] = {
    "4.6": "grok-4.6",
    "4.5": "grok-4.5",
}

# Grok has no documented output cap; keep a large safety limit.
GROK_MAX_OUTPUT_TOKENS = 128_000
GROK_DEFAULT_EFFORT = "high"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    thinking: dict | None = None
    output_config: dict | None = None
    max_tokens: int = FALLBACK_MAX_OUTPUT_TOKENS


def _resolve_api_id(family: str, version: str) -> str:
    return API_MODEL_IDS.get((family, version), f"claude-{family}-{version}")


def _thinking_config(model_id: str) -> dict:
    if model_id in ADAPTIVE_THINKING_MODELS:
        # Sonnet 5+ defaults display to "omitted", which returns empty thinking
        # blocks. Always request summarized text so transcripts and heartbeats work.
        return {"type": "adaptive", "display": "summarized"}
    return {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET, "display": "summarized"}


def _max_output_tokens(model_id: str) -> int:
    return MODEL_MAX_OUTPUT_TOKENS.get(model_id, FALLBACK_MAX_OUTPUT_TOKENS)


def _parse_grok_model_name(name: str) -> ModelSpec:
    """Parse grok-{version}[-{effort}] into API settings.

    Version may be dotted (`4.6`) or hyphenated (`4-6`). Grok always reasons;
    `-thinking` is accepted and ignored. Effort defaults to high.
    """
    parts = name.split("-")
    if len(parts) < 2 or parts[0] != "grok":
        raise ValueError(f"Invalid model name: {name!r}")

    rest = parts[1:]
    effort: str | None = None
    while rest:
        token = rest[-1]
        if token in GROK_EFFORT_LEVELS:
            effort = token
            rest = rest[:-1]
        elif token == "thinking":
            rest = rest[:-1]
        elif token == "max":
            raise ValueError(f"Model {name!r} does not support effort 'max'")
        else:
            break
    if not rest:
        raise ValueError(f"Missing version in model name: {name!r}")

    version_raw = "-".join(rest)
    if re.fullmatch(r"\d+-\d+", version_raw):
        version = version_raw.replace("-", ".", 1)
    else:
        version = version_raw
    model_id = GROK_API_MODEL_IDS.get(version, f"grok-{version}")
    effort = effort or GROK_DEFAULT_EFFORT
    return ModelSpec(
        name=name,
        provider="xai",
        model_id=model_id,
        thinking=None,
        output_config={"effort": effort},
        max_tokens=GROK_MAX_OUTPUT_TOKENS,
    )


def parse_model_name(name: str) -> ModelSpec:
    """Parse a friendly model name into API settings.

    Claude: claude-{family}-{version}[-thinking][-{effort}]
    Grok: grok-{version}[-thinking][-{effort}]
    """
    if name.startswith("grok-"):
        return _parse_grok_model_name(name)
    if not name.startswith("claude-"):
        raise ValueError(f"Model name must start with 'claude-' or 'grok-': {name!r}")

    parts = name.split("-")
    if len(parts) < 3 or parts[0] != "claude":
        raise ValueError(f"Invalid model name: {name!r}")

    family = parts[1]
    if family not in FAMILIES:
        raise ValueError(f"Unknown model family {family!r} in {name!r}. Expected one of: {sorted(FAMILIES)}")

    rest = parts[2:]
    thinking = False
    effort: str | None = None

    while rest:
        token = rest[-1]
        if token in EFFORT_LEVELS:
            effort = token
            rest = rest[:-1]
        elif token == "thinking":
            thinking = True
            rest = rest[:-1]
        else:
            break

    if not rest:
        raise ValueError(f"Missing version in model name: {name!r}")

    version = "-".join(rest)
    model_id = _resolve_api_id(family, version)

    thinking_config = _thinking_config(model_id) if thinking else None
    output_config = {"effort": effort} if effort else None

    if effort and model_id not in EFFORT_MODELS:
        raise ValueError(f"Model {model_id!r} does not support effort (from {name!r})")

    return ModelSpec(
        name=name,
        provider="anthropic",
        model_id=model_id,
        thinking=thinking_config,
        output_config=output_config,
        max_tokens=_max_output_tokens(model_id),
    )


def safe_result_dir_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)
