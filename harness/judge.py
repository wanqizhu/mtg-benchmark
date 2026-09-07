from __future__ import annotations

from typing import Any, Protocol

import anthropic
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from harness.config import get_api_key
from harness.model_names import ModelSpec
from harness.providers.anthropic import usage_from_response
from harness.providers.openai import (
    REQUEST_TIMEOUT_SECONDS,
    response_content_blocks,
    usage_from_openai,
)


def final_text(response: Any) -> str:
    text_blocks = [block.text for block in response.content if block.type == "text"]
    if not text_blocks:
        raise ValueError("No text block in judge response")
    return text_blocks[-1]


def final_text_from_openai(response: Any) -> str:
    text_blocks = [
        block["text"]
        for block in response_content_blocks(response)
        if block.get("type") == "text" and block.get("text")
    ]
    if not text_blocks:
        raise ValueError("No text block in judge response")
    return "\n".join(text_blocks)


class Judge(Protocol):
    def judge(self, *, system: str, user: str) -> tuple[str, dict[str, int]]: ...


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


class OpenAIJudge:
    def __init__(self, spec: ModelSpec, api_key: str | None = None) -> None:
        self._spec = spec
        self._client = OpenAI(
            api_key=api_key or get_api_key("openai"),
            timeout=float(REQUEST_TIMEOUT_SECONDS),
            max_retries=2,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _create(self, request: dict[str, Any]) -> Any:
        with self._client.responses.stream(**request) as stream:
            for event in stream:
                if getattr(event, "type", None) == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise RuntimeError(f"OpenAI judge response failed: {error}")
            response = stream.get_final_response()
        if getattr(response, "status", None) == "failed":
            raise RuntimeError(f"OpenAI judge response failed: {getattr(response, 'error', None)}")
        return response

    def judge(self, *, system: str, user: str) -> tuple[str, dict[str, int]]:
        reasoning = {
            "effort": str((self._spec.output_config or {}).get("effort") or "medium"),
            **(self._spec.thinking or {}),
        }
        request: dict[str, Any] = {
            "model": self._spec.model_id,
            "instructions": system,
            "input": [{"role": "user", "content": user}],
            "reasoning": reasoning,
            "max_output_tokens": self._spec.max_tokens,
            "store": False,
        }
        response = self._create(request)
        return final_text_from_openai(response), usage_from_openai(response.usage)


def make_judge(spec: ModelSpec) -> Judge:
    if spec.provider == "openai":
        return OpenAIJudge(spec)
    if spec.provider == "anthropic":
        return AnthropicJudge(spec)
    raise ValueError(f"No judge implementation for provider {spec.provider!r}")
