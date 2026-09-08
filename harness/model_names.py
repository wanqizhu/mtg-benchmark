from __future__ import annotations

import re
from dataclasses import dataclass

FAMILIES = frozenset({"haiku", "sonnet", "opus", "fable"})
EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
GROK_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh"})
OPENAI_EFFORT_LEVELS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})

# (family, version) -> Anthropic API model id
API_MODEL_IDS: dict[tuple[str, str], str] = {
    ("haiku", "4-5"): "claude-haiku-4-5-20251001",
    ("sonnet", "5"): "claude-sonnet-5",
    ("sonnet", "4-6"): "claude-sonnet-4-6",
    ("sonnet", "4-5"): "claude-sonnet-4-5-20250929",
    ("opus", "5"): "claude-opus-5",
    ("opus", "4-8"): "claude-opus-4-8",
    ("opus", "4-7"): "claude-opus-4-7",
    ("opus", "4-6"): "claude-opus-4-6",
    ("opus", "4-5"): "claude-opus-4-5-20251101",
    ("fable", "5"): "claude-fable-5",
    ("fable", "5-1"): "claude-fable-5-1",
}

# Models that use adaptive thinking when -thinking is set.
ADAPTIVE_THINKING_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-opus-5",
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-opus-4-8",
    }
)

# Models that support output_config.effort.
EFFORT_MODELS = frozenset(
    {
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5-20250929",
        "claude-opus-5",
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
    "claude-fable-5-1": 128_000,
    "claude-fable-5": 128_000,
    "claude-sonnet-5": 128_000,
    "claude-sonnet-4-6": 128_000,
    "claude-opus-5": 128_000,
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

# OpenAI Responses API friendly names use
# gpt-{version}[-{tier}][-thinking][-{effort}].
OPENAI_MAX_OUTPUT_TOKENS = 128_000
OPENAI_DEFAULT_EFFORT = "medium"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    thinking: dict | None = None
    output_config: dict | None = None
    max_tokens: int = FALLBACK_MAX_OUTPUT_TOKENS


def _normalize_claude_version(version: str) -> str:
    if re.fullmatch(r"\d+\.\d+", version):
        return version.replace(".", "-", 1)
    return version


def _resolve_api_id(family: str, version: str) -> str:
    version = _normalize_claude_version(version)
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


def _parse_openai_model_name(name: str) -> ModelSpec:
    """Parse GPT Responses API model names and reasoning settings."""
    parts = name.split("-")
    if len(parts) < 2 or parts[0] != "gpt":
        raise ValueError(f"Invalid model name: {name!r}")

    rest = parts[1:]
    effort: str | None = None
    thinking = False
    while rest:
        token = rest[-1]
        if token in OPENAI_EFFORT_LEVELS:
            effort = token
            rest = rest[:-1]
        elif token == "thinking":
            thinking = True
            rest = rest[:-1]
        else:
            break
    if not rest:
        raise ValueError(f"Missing version in model name: {name!r}")

    model_id = f"gpt-{'-'.join(rest)}"
    return ModelSpec(
        name=name,
        provider="openai",
        model_id=model_id,
        # OpenAI does not expose raw chain-of-thought. `summary=auto`
        # requests the most detailed reasoning summary the model supports.
        thinking={"summary": "auto", "context": "all_turns"} if thinking else None,
        output_config={"effort": effort or OPENAI_DEFAULT_EFFORT},
        max_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )


def parse_model_name(name: str) -> ModelSpec:
    """Parse a friendly model name into API settings.

    Claude: claude-{family}-{version}[-thinking][-{effort}]
    Grok: grok-{version}[-thinking][-{effort}]
    OpenAI: gpt-{version}[-{tier}][-thinking][-{effort}]
    """
    if name.startswith("grok-"):
        return _parse_grok_model_name(name)
    if name.startswith("gpt-"):
        return _parse_openai_model_name(name)
    if not name.startswith("claude-"):
        raise ValueError(
            f"Model name must start with 'claude-', 'grok-', or 'gpt-': {name!r}"
        )

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


_DISPLAY_EFFORTS = EFFORT_LEVELS | GROK_EFFORT_LEVELS | OPENAI_EFFORT_LEVELS
_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def _title_token(token: str) -> str:
    if token.lower() == "gpt":
        return "GPT"
    if not token:
        return token
    return token[:1].upper() + token[1:]


def friendly_model_name(name: str) -> str:
    """Short site label: 'Fable 5.1', dropping default thinking/high suffixes."""
    text = str(name or "").strip()
    if " @ " in text:
        left, _, right = text.partition(" @ ")
        return f"{friendly_model_name(left)} @ {right}"

    parts = text.split("-")
    effort: str | None = None
    while parts:
        token = parts[-1]
        if token in _DISPLAY_EFFORTS:
            effort = token
            parts.pop()
        elif token == "thinking":
            parts.pop()
        else:
            break

    if parts and parts[0] == "claude":
        parts = parts[1:]

    tokens: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if _VERSION_TOKEN.fullmatch(part):
            nums = [part]
            index += 1
            while index < len(parts) and parts[index].isdigit():
                nums.append(parts[index])
                index += 1
            tokens.append(".".join(nums))
        else:
            tokens.append(_title_token(part))
            index += 1

    label = " ".join(tokens) if tokens else text
    if effort and effort != "high":
        label = f"{label} {_title_token(effort)}".strip()
    return label
