from __future__ import annotations

from typing import Any

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from harness.config import get_api_key
from harness.model_names import ModelSpec
from harness.providers.anthropic import usage_from_response


def final_text(response: Any) -> str:
    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        raise ValueError("No text block in judge response")
    return text_blocks[-1]


class AnthropicJudge:
    def __init__(self, spec: ModelSpec, api_key: str | None = None) -> None:
        self._spec = spec
        self._client = anthropic.Anthropic(api_key=api_key or get_api_key("anthropic"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _create(self, **kwargs: Any) -> Any:
        thinking = kwargs.get("thinking")
        max_tokens = int(kwargs.get("max_tokens", 4096))
        if thinking or max_tokens > 8192:
            with self._client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()
        return self._client.messages.create(**kwargs)

    def judge(self, *, system: str, user: str) -> tuple[str, dict[str, int]]:
        request: dict[str, Any] = {
            "model": self._spec.model_id,
            "max_tokens": self._spec.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._spec.thinking:
            request["thinking"] = self._spec.thinking
        if self._spec.output_config:
            request["output_config"] = self._spec.output_config

        response = self._create(**request)
        return final_text(response), usage_from_response(response)
