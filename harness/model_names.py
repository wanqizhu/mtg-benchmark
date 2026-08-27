from __future__ import annotations

import re
from dataclasses import dataclass

FAMILIES = frozenset({"haiku", "sonnet", "opus", "fable"})
EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})

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


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    thinking: dict | None = None
    output_config: dict | None = None
    max_tokens: int = 4096


def _resolve_api_id(family: str, version: str) -> str:
    return API_MODEL_IDS.get((family, version), f"claude-{family}-{version}")


def _thinking_config(model_id: str) -> dict:
    if model_id in ADAPTIVE_THINKING_MODELS:
        # Sonnet 5+ defaults display to "omitted", which returns empty thinking
        # blocks. Always request summarized text so transcripts and heartbeats work.
        return {"type": "adaptive", "display": "summarized"}
    return {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET, "display": "summarized"}


def _default_max_tokens(*, thinking: bool, effort: str | None) -> int:
    if thinking or effort in {"xhigh", "max"}:
        return 64_000
    if effort in {"high", "medium"}:
        return 16_000
    return 4096


def parse_model_name(name: str) -> ModelSpec:
    """Parse claude-{family}-{version}[-thinking][-{effort}] into API settings."""
    if not name.startswith("claude-"):
        raise ValueError(f"Model name must start with 'claude-': {name!r}")

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
        max_tokens=_default_max_tokens(thinking=thinking, effort=effort),
    )


def safe_result_dir_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)
