"""OpenRouter: open-weight models over OpenAI-style chat completions.

Shares the streaming tool loop with the xAI provider. Differences:
  - reasoning is requested via `reasoning: {"effort": ...}`;
  - usage (including billed USD `cost`) is requested via `usage: {"include": true}`;
  - reasoning text streams as `delta.reasoning`, with `reasoning_details` blocks
    that must be echoed back unchanged on tool turns.
https://openrouter.ai/docs/api_reference/overview
"""

from __future__ import annotations

from typing import Any

from harness.core import Tool
from harness.model_names import ModelSpec
from harness.providers.xai import XAIProvider, _tool_defs

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CACHE_TTL = "auto"

# Pin each model to a fixed host so every turn (and every puzzle) runs on the
# same weights at the same precision. Unpinned, OpenRouter load-balances across
# hosts by price, many of which serve fp4 quants, and a host switch mid-rollout
# also defeats prompt caching. Prefer the lab's own endpoint at native
# precision; where the lab doesn't host, pick an fp8/bf16 endpoint. Entries are
# ordered fallbacks; `allow_fallbacks: false` keeps routing inside the list.
# Slugs come from https://openrouter.ai/api/v1/models/{id}/endpoints (`tag`).
OPENROUTER_PROVIDER_PINS: dict[str, list[str]] = {
    "z-ai/glm-5.3": ["z-ai/fp8"],
    "z-ai/glm-5.2": ["z-ai/fp8"],
    # DeepSeek V4 trains in fp8; no first-party endpoint on OpenRouter for these.
    "deepseek/deepseek-v4-pro": ["deepinfra/fp8", "baidu/fp8"],
    "deepseek/deepseek-v4-flash": ["deepinfra/fp8", "baidu/fp8"],
    "deepseek/deepseek-v4.1-flash": ["deepseek"],
    "moonshotai/kimi-k3": ["moonshotai/mxfp4"],
    "moonshotai/kimi-k2.6": ["moonshotai/int4"],
    "qwen/qwen3.8-2.4t-a95b": ["alibaba"],
    "qwen/qwen3.8-max-0902": ["alibaba"],
    "qwen/qwen3.7-max": ["alibaba"],
    "nvidia/nemotron-3-ultra-550b-a55b": ["venice/fp8", "baseten/fp4"],
    "minimax/minimax-m3": ["minimax/fp8"],
    # gpt-oss ships in mxfp4; any host is native precision.
    "openai/gpt-oss-120b": ["groq", "deepinfra/turbo"],
    "openai/gpt-oss-20b": ["groq"],
    "google/gemma-4-31b-it": ["crusoe/bf16", "deepinfra/fp8"],
}


def provider_pin(model_id: str) -> list[str] | None:
    pins = OPENROUTER_PROVIDER_PINS.get(model_id)
    return list(pins) if pins else None


class OpenRouterProvider(XAIProvider):
    provider_name = "openrouter"
    base_url = OPENROUTER_BASE_URL
    label = "OpenRouter"
    reasoning_delta_key = "reasoning"

    def _config_extra(self, spec: ModelSpec) -> dict[str, Any]:
        pins = provider_pin(spec.model_id)
        return {"provider_pin": pins} if pins else {}

    def _static_headers(self) -> dict[str, str]:
        # Optional attribution headers; OpenRouter uses them for its app rankings.
        return {
            "HTTP-Referer": "https://github.com/wanqizhu/mtg-benchmark",
            "X-Title": "mtg-benchmark",
        }

    def _request_headers(self, conv_id: str) -> dict[str, str]:
        return {}

    def _request_body(
        self,
        *,
        spec: ModelSpec,
        messages: list[dict[str, Any]],
        tools: list[Tool],
        effort: str,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": spec.model_id,
            "messages": messages,
            "max_tokens": spec.max_tokens,
            "reasoning": {"effort": effort},
            "stream": True,
            "usage": {"include": True},
        }
        pins = provider_pin(spec.model_id)
        if pins:
            body["provider"] = {"order": pins, "allow_fallbacks": False}
        if tools:
            body["tools"] = _tool_defs(tools)
        return body
