from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Anthropic API list prices (USD per million tokens).
# Source: https://platform.claude.com/docs/en/about-claude/pricing
PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_5m_per_mtok: float
    cache_write_1h_per_mtok: float


MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-fable-5": ModelPricing(10, 50, 1, 12.5, 20),
    "claude-opus-4-8": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-7": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-6": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-opus-4-5-20251101": ModelPricing(5, 25, 0.5, 6.25, 10),
    "claude-sonnet-5": ModelPricing(2, 10, 0.2, 2.5, 4),
    "claude-sonnet-4-6": ModelPricing(3, 15, 0.3, 3.75, 6),
    "claude-sonnet-4-5-20250929": ModelPricing(3, 15, 0.3, 3.75, 6),
    "claude-haiku-4-5-20251001": ModelPricing(1, 5, 0.1, 1.25, 2),
}


def _mtok_cost(tokens: int, rate_per_mtok: float) -> float:
    return tokens * rate_per_mtok / 1_000_000


def get_model_pricing(model_id: str) -> ModelPricing:
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        raise KeyError(f"No pricing configured for model {model_id!r}")
    return pricing


def estimate_cost(
    usage: dict[str, int],
    *,
    model_id: str,
    cache_ttl: str = "5m",
) -> dict[str, Any]:
    """Estimate USD cost from Anthropic usage fields."""
    if not usage:
        return {"usd": 0.0, "breakdown_usd": {}}

    pricing = get_model_pricing(model_id)
    cache_write_rate = (
        pricing.cache_write_1h_per_mtok if cache_ttl == "1h" else pricing.cache_write_5m_per_mtok
    )

    breakdown = {
        "input": _mtok_cost(usage.get("input_tokens", 0), pricing.input_per_mtok),
        "output": _mtok_cost(usage.get("output_tokens", 0), pricing.output_per_mtok),
        "cache_creation": _mtok_cost(
            usage.get("cache_creation_input_tokens", 0), cache_write_rate
        ),
        "cache_read": _mtok_cost(
            usage.get("cache_read_input_tokens", 0), pricing.cache_read_per_mtok
        ),
    }
    total = sum(breakdown.values())
    return {
        "usd": round(total, 6),
        "breakdown_usd": {key: round(value, 6) for key, value in breakdown.items()},
        "model_id": model_id,
        "cache_ttl": cache_ttl if usage.get("cache_creation_input_tokens") else None,
        "pricing_source": PRICING_SOURCE,
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

    if result_path and "inline-rules" in result_path:
        return "1h"
    if result_path and "tools-rules" in result_path:
        return "5m"
    return "5m"
