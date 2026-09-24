from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator

import httpx

from harness.config import DEFAULT_MAX_TOKEN_CONTINUES, get_api_key
from harness.core import Tool, Transcript
from harness.model_names import ModelSpec
from harness.providers.base import append_max_token_continue
from harness.progress import RolloutProgress
from harness.prompt_log import log_tools

XAI_BASE_URL = "https://api.x.ai/v1"
STREAM_STALL_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 3600
# OpenRouter closes an upstream that emits no tokens for ~190s during long
# reasoning. Retry the same turn; the request is idempotent until we persist it.
STREAM_RETRY_ATTEMPTS = 5
STREAM_RETRY_DELAY_SECONDS = 8
CACHE_TTL = "auto"
STOP_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


def _tool_defs(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def _tool_map(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def _reasoning_effort(spec: ModelSpec) -> str:
    config = spec.output_config or {}
    return str(config.get("effort") or "high")


def usage_from_xai(usage: Any) -> dict[str, Any]:
    """Normalize xAI/OpenAI usage onto Anthropic-shaped fields."""
    if not usage:
        return {}
    if not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "prompt_tokens_details": getattr(usage, "prompt_tokens_details", None),
            "completion_tokens_details": getattr(usage, "completion_tokens_details", None),
        }
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        details = {"cached_tokens": getattr(details, "cached_tokens", 0)}
    cached = int(details.get("cached_tokens") or 0)
    completion_details = usage.get("completion_tokens_details") or {}
    if not isinstance(completion_details, dict):
        completion_details = {
            "reasoning_tokens": getattr(completion_details, "reasoning_tokens", 0)
        }
    reasoning = int(completion_details.get("reasoning_tokens") or 0)
    # OpenRouter reports explicit cache writes for providers that bill them.
    cache_write = int(details.get("cache_write_tokens") or 0)
    # xAI bills reasoning at the output rate. completion_tokens is often visible
    # text only (a short reply can report completion=1 with hundreds of reasoning tokens).
    output = completion + reasoning if reasoning and completion < reasoning else completion
    normalized: dict[str, Any] = {
        "input_tokens": max(0, prompt - cached - cache_write),
        "output_tokens": output,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cached,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
    }
    if reasoning:
        normalized["thinking_tokens"] = reasoning
    # OpenRouter returns the billed USD amount; pricing.py prefers it when the
    # model has no rate table.
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if cost is not None:
        normalized["reported_cost_usd"] = float(cost)
    return normalized


def _merge_usage(total: dict[str, Any], turn_usage: dict[str, Any]) -> None:
    for key, value in turn_usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            existing = total.get(key, 0)
            total[key] = existing + value if isinstance(existing, (int, float)) else value
        elif isinstance(value, dict):
            nested = total.get(key)
            if not isinstance(nested, dict):
                nested = {}
                total[key] = nested
            _merge_usage(nested, value)
        else:
            total[key] = value


def assistant_content_blocks(api_message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    # xAI streams `reasoning_content`; OpenRouter streams `reasoning`.
    reasoning = api_message.get("reasoning_content") or api_message.get("reasoning")
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
    text = api_message.get("content")
    if text:
        blocks.append({"type": "text", "text": text})
    for call in api_message.get("tool_calls") or []:
        arguments = ((call.get("function") or {}).get("arguments")) or "{}"
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"_raw": arguments}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id"),
                "name": (call.get("function") or {}).get("name"),
                "input": parsed,
            }
        )
    return blocks


def parse_tool_call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = ((call.get("function") or {}).get("arguments")) or "{}"
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid tool arguments for {call.get('id')}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Tool arguments must be an object, got {type(parsed).__name__}")
    return parsed


def map_stop_reason(finish_reason: str | None) -> str | None:
    if not finish_reason:
        return finish_reason
    return STOP_REASON_MAP.get(finish_reason, finish_reason)


def _config_value(turns: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    for turn in turns:
        if turn.get("role") == "config" and key in turn:
            return turn[key]
    return default


def messages_from_transcript(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        api_messages = turn.get("api_messages")
        if isinstance(api_messages, list):
            messages.extend(item for item in api_messages if isinstance(item, dict))
            continue
        api_message = turn.get("api_message")
        if isinstance(api_message, dict):
            messages.append(api_message)
            continue
        role = turn.get("role")
        if role == "user":
            text = turn.get("text")
            if isinstance(text, str):
                messages.append({"role": "user", "content": text})
                continue
            content = turn.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id"),
                                "content": block.get("content") or "",
                            }
                        )
                    elif block.get("type") == "text" and block.get("text"):
                        messages.append({"role": "user", "content": block["text"]})
        elif role == "assistant":
            content = turn.get("content")
            reconstructed: dict[str, Any] = {"role": "assistant", "content": turn.get("text")}
            tool_calls = []
            if isinstance(content, list):
                texts: list[str] = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "thinking" and block.get("thinking"):
                        reconstructed["reasoning_content"] = block["thinking"]
                    elif block.get("type") == "text" and block.get("text"):
                        texts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id"),
                                "type": "function",
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input") or {}),
                                },
                            }
                        )
                if texts:
                    reconstructed["content"] = "\n".join(texts)
            if tool_calls:
                reconstructed["tool_calls"] = tool_calls
            messages.append(reconstructed)
        elif role == "system" and turn.get("text"):
            messages.append({"role": "system", "content": turn["text"]})
    return messages


def queue_max_token_continue(
    transcript_turns: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    count: int,
    limit: int,
) -> int:
    """Append the same ``continue`` user turn a live max_tokens stop would.

    Resume loads a rollout that already stopped on the cap, so the continue
    has to be queued before the next request. Returns the updated count.
    """
    continue_message = {"role": "user", "content": "continue"}
    updated = append_max_token_continue(
        transcript_turns,
        count,
        limit,
        api_message=continue_message,
    )
    if updated != count:
        messages.append(continue_message)
    return updated


def _is_retryable_stream_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            httpx.RemoteProtocolError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectError,
        ),
    ):
        return True
    text = str(exc).lower()
    return (
        "idle timeout" in text
        or "incomplete chunked read" in text
        or "handshake operation timed out" in text
    )


def _merge_reasoning_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce streamed reasoning_details chunks by (type, id, index).

    OpenRouter streams the same detail across many deltas with partial `text`
    or `summary`; the model expects one entry per block on the next request.
    """
    merged: list[dict[str, Any]] = []
    positions: dict[tuple[Any, Any, Any], int] = {}
    for detail in details:
        key = (detail.get("type"), detail.get("id"), detail.get("index"))
        if key in positions:
            target = merged[positions[key]]
            for field in ("text", "summary", "data"):
                if isinstance(detail.get(field), str):
                    target[field] = (target.get(field) or "") + detail[field]
            for field, value in detail.items():
                if field not in ("text", "summary", "data") and value is not None:
                    target[field] = value
        else:
            positions[key] = len(merged)
            merged.append(dict(detail))
    return merged


def _iter_sse_json(response: httpx.Response) -> Iterator[dict[str, Any]]:
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


class XAIProvider:
    """OpenAI-style streaming chat completions with tools and reasoning.

    Subclasses (OpenRouter) override the class attributes and the request hooks.
    """

    provider_name = "xai"
    base_url = XAI_BASE_URL
    label = "xAI"
    # Streaming delta field carrying reasoning text.
    reasoning_delta_key = "reasoning_content"

    def __init__(self, api_key: str | None = None) -> None:
        timeout = httpx.Timeout(
            connect=10.0,
            read=REQUEST_TIMEOUT_SECONDS,
            write=30.0,
            pool=10.0,
        )
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key or get_api_key(self.provider_name)}",
                "Content-Type": "application/json",
                **self._static_headers(),
            },
            timeout=timeout,
        )

    def _static_headers(self) -> dict[str, str]:
        return {}

    def _config_extra(self, spec: ModelSpec) -> dict[str, Any]:
        """Provider-specific fields recorded on the transcript's config turn."""
        return {}

    def _request_headers(self, conv_id: str) -> dict[str, str]:
        return {"x-grok-conv-id": conv_id}

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
            "reasoning_effort": effort,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = _tool_defs(tools)
        return body

    def _create(
        self,
        *,
        progress: RolloutProgress | None = None,
        api_turn: int = 0,
        conv_id: str,
        body: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], str | None, dict[str, Any]]:
        if progress:
            progress.on_api_turn_start(api_turn)

        content = ""
        reasoning = ""
        reasoning_details: list[dict[str, Any]] = []
        tool_acc: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        usage: dict[str, Any] = {}
        # Routing facts about this turn (e.g. which host OpenRouter used).
        meta: dict[str, Any] = {}
        last_progress = time.monotonic()
        last_logged_chars = 0
        last_size = 0

        with self._client.stream(
            "POST",
            "/chat/completions",
            json=body,
            headers=self._request_headers(conv_id),
        ) as response:
            if response.status_code >= 400:
                error_body = response.read().decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"{self.label} chat completions failed with {response.status_code}: {error_body}",
                    request=response.request,
                    response=response,
                )
            for event in _iter_sse_json(response):
                now = time.monotonic()
                if event.get("error"):
                    # OpenRouter reports upstream failures as an in-stream event.
                    raise RuntimeError(f"{self.label} stream error: {event['error']}")
                if event.get("provider"):
                    meta["served_by"] = event["provider"]
                if event.get("usage"):
                    usage = usage_from_xai(event["usage"])
                    if progress:
                        progress.on_message_start(usage)
                        progress.update_stream(output_tokens=int(usage.get("output_tokens") or 0))
                    last_progress = now
                choices = event.get("choices") or []
                if not choices:
                    if now - last_progress > STREAM_STALL_TIMEOUT_SECONDS:
                        raise TimeoutError(
                            f"{self.label} stream stalled for {STREAM_STALL_TIMEOUT_SECONDS} seconds "
                            f"during API turn {api_turn}"
                        )
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                for detail in delta.get("reasoning_details") or []:
                    if isinstance(detail, dict):
                        reasoning_details.append(detail)
                reasoning_delta = delta.get(self.reasoning_delta_key) or ""
                if reasoning_delta:
                    reasoning += reasoning_delta
                    last_progress = now
                    if progress:
                        progress.update_stream(
                            block_type="thinking",
                            output_chars=len(reasoning) + len(content),
                        )
                text_delta = delta.get("content") or ""
                if text_delta:
                    content += text_delta
                    last_progress = now
                    if progress:
                        progress.update_stream(
                            block_type="text",
                            output_chars=len(reasoning) + len(content),
                        )
                for tool_delta in delta.get("tool_calls") or []:
                    index = int(tool_delta.get("index") or 0)
                    entry = tool_acc.setdefault(index, {"id": "", "name": "", "arguments": ""})
                    if tool_delta.get("id"):
                        entry["id"] = tool_delta["id"]
                    function = tool_delta.get("function") or {}
                    if function.get("name"):
                        entry["name"] += function["name"]
                    if function.get("arguments"):
                        entry["arguments"] += function["arguments"]
                    last_progress = now
                    if progress:
                        progress.update_stream(
                            block_type="tool_use",
                            output_chars=len(reasoning)
                            + len(content)
                            + sum(len(item["arguments"]) for item in tool_acc.values()),
                        )
                visible = len(reasoning) + len(content)
                if visible > last_size:
                    last_size = visible
                    last_progress = now
                if now - last_progress > STREAM_STALL_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"{self.label} stream stalled for {STREAM_STALL_TIMEOUT_SECONDS} seconds "
                        f"during API turn {api_turn}"
                    )
                if progress and visible - last_logged_chars >= 25_000:
                    last_logged_chars = visible
                    block = "thinking" if reasoning and not content else "text" if content else "tool_use"
                    progress.log(
                        f"api turn {api_turn} streaming ({block}, {visible:,} output chars so far)"
                    )

        tool_calls = []
        for index in sorted(tool_acc):
            entry = tool_acc[index]
            if not entry["id"] or not entry["name"]:
                continue
            tool_calls.append(
                {
                    "id": entry["id"],
                    "type": "function",
                    "function": {"name": entry["name"], "arguments": entry["arguments"] or "{}"},
                }
            )
        api_message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if reasoning:
            api_message[self.reasoning_delta_key] = reasoning
        if reasoning_details:
            # Must be echoed back unmodified on tool turns (encrypted/summarized reasoning).
            api_message["reasoning_details"] = _merge_reasoning_details(reasoning_details)
        if tool_calls:
            api_message["tool_calls"] = tool_calls
        return api_message, usage, finish_reason, meta

    def run_rollout(
        self,
        *,
        spec: ModelSpec,
        system: str,
        prompt: str,
        tools: list[Tool],
        max_turns: int,
        max_token_continues: int = DEFAULT_MAX_TOKEN_CONTINUES,
        resume_from: Transcript | None = None,
        progress: RolloutProgress | None = None,
    ) -> Transcript:
        tool_by_name = _tool_map(tools)
        effort = _reasoning_effort(spec)

        if resume_from is None:
            conv_id = str(uuid.uuid4())
            transcript_turns: list[dict[str, Any]] = [
                {"role": "system", "text": system},
                {"role": "tools", "tools": log_tools(tools)},
                {
                    "role": "config",
                    "model": spec.name,
                    "model_id": spec.model_id,
                    "thinking": spec.thinking,
                    "output_config": spec.output_config,
                    "max_tokens": spec.max_tokens,
                    "conv_id": conv_id,
                    "cache_ttl": CACHE_TTL,
                    **self._config_extra(spec),
                },
                {"role": "user", "text": prompt, "api_message": {"role": "user", "content": prompt}},
            ]
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            usage: dict[str, Any] = {}
            tool_call_count = 0
        else:
            transcript_turns = list(resume_from.turns)
            conv_id = _config_value(transcript_turns, "conv_id") or str(uuid.uuid4())
            messages = messages_from_transcript(transcript_turns)
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": system}, *messages]
            usage = dict(resume_from.usage)
            tool_call_count = resume_from.tool_call_count

        max_token_continue_count = sum(
            1
            for turn in transcript_turns
            if turn.get("role") == "user" and turn.get("reason") == "continue_after_max_tokens"
        )
        if resume_from is not None:
            queued = max_token_continue_count
            max_token_continue_count = queue_max_token_continue(
                transcript_turns,
                messages,
                max_token_continue_count,
                max_token_continues,
            )
            if max_token_continue_count != queued and progress:
                progress.log(
                    f"continuing after max_tokens "
                    f"({max_token_continue_count}/{max_token_continues})"
                )

        if progress:
            progress.write(
                Transcript(
                    turns=transcript_turns,
                    tool_call_count=tool_call_count,
                    usage=usage,
                    model=spec.model_id,
                    provider=self.provider_name,
                ),
                status="in_progress",
            )
            progress.log("rollout resumed" if resume_from is not None else "rollout started")

        max_api_turns = max_turns + max_token_continues
        while sum(1 for turn in transcript_turns if turn.get("role") == "assistant") < max_api_turns:
            api_turn = sum(1 for turn in transcript_turns if turn.get("role") == "assistant") + 1
            body = self._request_body(spec=spec, messages=messages, tools=tools, effort=effort)

            turn_started = time.monotonic()
            turn_started_at = datetime.now(timezone.utc).isoformat()
            api_message: dict[str, Any] | None = None
            turn_usage: dict[str, Any] = {}
            finish_reason: str | None = None
            meta: dict[str, Any] = {}
            last_exc: BaseException | None = None
            for attempt in range(1, STREAM_RETRY_ATTEMPTS + 1):
                try:
                    api_message, turn_usage, finish_reason, meta = self._create(
                        progress=progress,
                        api_turn=api_turn,
                        conv_id=conv_id,
                        body=body,
                    )
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt == STREAM_RETRY_ATTEMPTS or not _is_retryable_stream_error(exc):
                        raise
                    if progress:
                        progress.log(
                            f"api turn {api_turn} retry {attempt}/{STREAM_RETRY_ATTEMPTS - 1} "
                            f"after {type(exc).__name__}: {exc}"
                        )
                    time.sleep(STREAM_RETRY_DELAY_SECONDS)
            if last_exc is not None or api_message is None:
                raise last_exc or RuntimeError("stream retry exhausted without a response")
            turn_duration_ms = int((time.monotonic() - turn_started) * 1000)
            turn_finished_at = datetime.now(timezone.utc).isoformat()
            _merge_usage(usage, turn_usage)
            stop_reason = map_stop_reason(finish_reason)
            assistant_content = assistant_content_blocks(api_message)
            text_parts = [
                block["text"] for block in assistant_content if block.get("type") == "text"
            ]
            transcript_turns.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "text": "\n".join(text_parts) if text_parts else None,
                    "stop_reason": stop_reason,
                    "usage": turn_usage,
                    "started_at": turn_started_at,
                    "finished_at": turn_finished_at,
                    "duration_ms": turn_duration_ms,
                    "api_message": api_message,
                    **meta,
                }
            )
            messages.append(api_message)

            if stop_reason == "max_tokens" and max_token_continue_count < max_token_continues:
                max_token_continue_count += 1
                continue_message = {"role": "user", "content": "continue"}
                messages.append(continue_message)
                transcript_turns.append(
                    {
                        "role": "user",
                        "text": "continue",
                        "reason": "continue_after_max_tokens",
                        "api_message": continue_message,
                    }
                )
                if progress:
                    progress.on_turn_finished(
                        Transcript(
                            turns=transcript_turns,
                            tool_call_count=tool_call_count,
                            usage=usage,
                            model=spec.model_id,
                            provider=self.provider_name,
                        )
                    )
                    progress.log(
                        f"continuing after max_tokens "
                        f"({max_token_continue_count}/{max_token_continues})"
                    )
                continue

            tool_calls = api_message.get("tool_calls") or []
            if tool_calls:
                tool_result_blocks: list[dict[str, Any]] = []
                api_tool_messages: list[dict[str, Any]] = []
                for call in tool_calls:
                    tool_call_count += 1
                    name = (call.get("function") or {}).get("name")
                    tool = tool_by_name.get(name)
                    try:
                        arguments = parse_tool_call_arguments(call)
                        if tool is None:
                            result = f"Unknown tool: {name}"
                        else:
                            result = tool.run(**arguments)
                    except Exception as exc:  # noqa: BLE001
                        result = f"Tool error: {exc}"
                    tool_result = {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": result,
                    }
                    messages.append(tool_result)
                    api_tool_messages.append(tool_result)
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.get("id"),
                            "content": result,
                        }
                    )
                transcript_turns.append(
                    {
                        "role": "user",
                        "content": tool_result_blocks,
                        "api_messages": api_tool_messages,
                    }
                )
                if progress:
                    progress.on_turn_finished(
                        Transcript(
                            turns=transcript_turns,
                            tool_call_count=tool_call_count,
                            usage=usage,
                            model=spec.model_id,
                            provider=self.provider_name,
                        )
                    )
                continue

            if progress:
                progress.on_turn_finished(
                    Transcript(
                        turns=transcript_turns,
                        tool_call_count=tool_call_count,
                        usage=usage,
                        model=spec.model_id,
                        provider=self.provider_name,
                    )
                )
            break

        final = Transcript(
            turns=transcript_turns,
            tool_call_count=tool_call_count,
            usage=usage,
            model=spec.model_id,
            provider=self.provider_name,
        )
        if progress:
            progress.on_rollout_complete(final)
        return final
