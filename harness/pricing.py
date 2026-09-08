from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Anthropic API list prices (USD per million tokens).
# Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
XAI_PRICING_SOURCE = "https://docs.x.ai/developers/pricing"
OPENAI_PRICING_SOURCE = "https://developers.openai.com/api/docs/models/gpt-5.6-sol"


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float
    long_context_threshold: int | None = None
    long_context_multiplier: float = 1.0
    long_context_output_multiplier: float | None = None
    pricing_source: str = PRICING_SOURCE


MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-fable-5-1": ModelPricing(10, 50, 0.25, 12.5, 20),
    "claude-fable-5": ModelPricing(10, 50, 1, 12.5, 20),
    "claude-opus-5": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-8": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-7": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-6": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-5-20251101": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-sonnet-5": ModelPricing(2, 10, 0.2, 2.5, 4),
    "claude-sonnet-4-6": ModelPricing(3, 15, 0.3, 3.75, 6),
    "claude-sonnet-4-5-20250929": ModelPricing(3, 15, 0.3, 3.75, 6),
    "claude-haiku-4-5-20251001": ModelPricing(1, 5, 0.1, 1.25, 2),
    # OpenAI output_tokens includes reasoning_tokens. Cache writes are billed
    # at 1.25x input. Above 272K prompt tokens, input/cache rates double while
    # output is 1.5x for the entire request.
    "gpt-5.6-sol": ModelPricing(
        4,
        20,
        0.4,
        5,
        5,
        long_context_threshold=272_001,
        long_context_multiplier=2.0,
        long_context_output_multiplier=1.5,
        pricing_source=OPENAI_PRICING_SOURCE,
    ),
    "gpt-5.6-terra": ModelPricing(
        2,
        12,
        0.2,
        2.5,
        2.5,
        long_context_threshold=272_001,
        long_context_multiplier=2.0,
        long_context_output_multiplier=1.5,
        pricing_source="https://developers.openai.com/api/docs/models/gpt-5.6-terra",
    ),
    "gpt-5.6-luna": ModelPricing(
        0.2,
        1.2,
        0.02,
        0.25,
        0.25,
        long_context_threshold=272_001,
        long_context_multiplier=2.0,
        long_context_output_multiplier=1.5,
        pricing_source="https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    ),
    "gpt-6-astra": ModelPricing(
        10,
        50,
        1,
        12.5,
        12.5,
        long_context_threshold=272_001,
        long_context_multiplier=2.0,
        long_context_output_multiplier=1.5,
        pricing_source="https://developers.openai.com/api/docs/models/gpt-6-astra",
    ),
    # Cached input is billed at cache_read; xAI has no separate cache-write fee.
    # Rates double when a request's prompt (including cached tokens) is >= 200k.
    "grok-4.6": ModelPricing(
        2,
        6,
        0.5,
        0,
        0,
        long_context_threshold=200_000,
        long_context_multiplier=2.0,
        pricing_source=XAI_PRICING_SOURCE,
    ),
    "grok-4.5": ModelPricing(
        2,
        6,
        0.3,
        0,
        0,
        long_context_threshold=200_000,
        long_context_multiplier=2.0,
        pricing_source=XAI_PRICING_SOURCE,
    ),
}


def _mtok_cost(tokens: int, rate_per_mtok: float) -> float:
    return tokens * rate_per_mtok / 1_000_000


def get_model_pricing(model_id: str) -> ModelPricing:
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        raise KeyError(f"No pricing configured for model {model_id!r}")
    return pricing


def prompt_token_count(usage: dict[str, Any] | None) -> int:
    """Uncached + cached input tokens for one request."""
    if not usage:
        return 0
    if usage.get("prompt_tokens"):
        return int(usage["prompt_tokens"])
    return (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )


def estimate_cost(
    usage: dict[str, int],
    *,
    model_id: str,
    cache_ttl: str = "1h",
) -> dict[str, Any]:
    """Estimate USD cost from normalized usage fields (Anthropic-shaped)."""
    if not usage:
        return {"usd": 0.0, "breakdown_usd": {}}

    pricing = get_model_pricing(model_id)
    multiplier = 1.0
    if pricing.long_context_threshold and prompt_token_count(usage) >= pricing.long_context_threshold:
        multiplier = pricing.long_context_multiplier
    output_multiplier = (
        pricing.long_context_output_multiplier
        if multiplier > 1 and pricing.long_context_output_multiplier is not None
        else multiplier
    )
    if cache_ttl == "1h":
        cache_write_rate = pricing.cache_write_1h_per_mtok
    elif cache_ttl in {"5m", "auto"}:
        cache_write_rate = pricing.cache_write_5m_per_mtok
    else:
        cache_write_rate = pricing.cache_write_5m_per_mtok
    cache_write_rate *= multiplier

    breakdown = {
        "input": _mtok_cost(usage.get("input_tokens", 0), pricing.input_per_mtok * multiplier),
        "output": _mtok_cost(
            usage.get("output_tokens", 0),
            pricing.output_per_mtok * output_multiplier,
        ),
        "cache_creation": _mtok_cost(
            usage.get("cache_creation_input_tokens", 0), cache_write_rate
        ),
        "cache_read": _mtok_cost(
            usage.get("cache_read_input_tokens", 0), pricing.cache_read_per_mtok * multiplier
        ),
    }
    total = sum(breakdown.values())
    return {
        "usd": round(total, 6),
        "breakdown_usd": {key: round(value, 6) for key, value in breakdown.items()},
        "model_id": model_id,
        "cache_ttl": cache_ttl if usage.get("cache_creation_input_tokens") else None,
        "pricing_source": pricing.pricing_source,
        "long_context": multiplier > 1,
    }


def estimate_rollout_cost(
    *,
    usage: dict[str, Any] | None,
    turns: list[dict[str, Any]] | None = None,
    model_id: str,
    cache_ttl: str = "1h",
) -> dict[str, Any]:
    """Sum per-turn cost when turn usage exists (needed for grok long-context tiers)."""
    turn_usages = [
        turn["usage"]
        for turn in turns or []
        if isinstance(turn, dict)
        and turn.get("role") == "assistant"
        and isinstance(turn.get("usage"), dict)
        and turn["usage"]
    ]
    if len(turn_usages) <= 1:
        return estimate_cost(usage or {}, model_id=model_id, cache_ttl=cache_ttl)

    breakdown = {"input": 0.0, "output": 0.0, "cache_creation": 0.0, "cache_read": 0.0}
    total = 0.0
    last: dict[str, Any] = {}
    long_context = False
    for turn_usage in turn_usages:
        last = estimate_cost(turn_usage, model_id=model_id, cache_ttl=cache_ttl)
        total += float(last["usd"])
        long_context = long_context or bool(last.get("long_context"))
        for key, value in last.get("breakdown_usd", {}).items():
            breakdown[key] = breakdown.get(key, 0.0) + float(value)
    return {
        "usd": round(total, 6),
        "breakdown_usd": {key: round(value, 6) for key, value in breakdown.items()},
        "model_id": model_id,
        "cache_ttl": last.get("cache_ttl"),
        "pricing_source": last.get("pricing_source", PRICING_SOURCE),
        "long_context": long_context,
    }


def infer_cache_ttl(payload: dict[str, Any], *, result_path: str | None = None) -> str:
    ttl = payload.get("cache_ttl")
    if ttl in {"5m", "1h"}:
        return ttl

    rules_mode = payload.get("rules_mode") or payload.get("reference", {}).get("rules_mode")
    if rules_mode == "inline":
        return "1h"
    if rules_mode == "tools":
        return "5m"

    if result_path and ("full-rules-in-context" in result_path or "inline-rules" in result_path):
        return "1h"
    if result_path and ("grep-rules" in result_path or "tools-rules" in result_path):
        return "5m"
    return "5m"


def reclassify_cache_as_input(usage: dict[str, Any]) -> int:
    """Move cache write/read tokens onto uncached input. Returns tokens moved."""
    if not usage:
        return 0
    moved = 0
    for key in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        tokens = int(usage.get(key) or 0)
        if not tokens:
            continue
        usage[key] = 0
        moved += tokens
    nested = usage.get("cache_creation")
    if isinstance(nested, dict):
        for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
            if nested.get(key):
                nested[key] = 0
    if moved:
        usage["input_tokens"] = int(usage.get("input_tokens") or 0) + moved
    return moved


def reclassify_tools_result_cache_as_input(payload: dict[str, Any]) -> int:
    """Rewrite a tools-mode result as if nothing was cached. Returns tokens moved."""
    transcript = payload.get("transcript")
    if not isinstance(transcript, dict):
        return 0
    moved = 0
    usage = transcript.get("usage")
    if isinstance(usage, dict):
        moved += reclassify_cache_as_input(usage)
    for turn in transcript.get("turns") or []:
        if isinstance(turn, dict) and isinstance(turn.get("usage"), dict):
            reclassify_cache_as_input(turn["usage"])
    model_id = payload.get("model_id")
    if moved and isinstance(usage, dict) and model_id:
        payload["cost"] = estimate_rollout_cost(
            usage=usage,
            turns=transcript.get("turns"),
            model_id=model_id,
        )
    return moved


def reclassify_1h_cache_write_as_read(usage: dict[str, Any]) -> int:
    """Move 1h cache-creation tokens onto cache-read. Returns tokens moved."""
    if not usage:
        return 0
    nested = usage.get("cache_creation")
    moved = 0
    if isinstance(nested, dict):
        moved = int(nested.get("ephemeral_1h_input_tokens") or 0)
        if moved:
            nested["ephemeral_1h_input_tokens"] = 0
    if not moved:
        return 0
    usage["cache_creation_input_tokens"] = max(
        0, int(usage.get("cache_creation_input_tokens") or 0) - moved
    )
    usage["cache_read_input_tokens"] = int(usage.get("cache_read_input_tokens") or 0) + moved
    return moved
