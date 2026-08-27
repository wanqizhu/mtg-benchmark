from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import anthropic
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from harness.config import get_api_key
from harness.core import Tool, Transcript
from harness.model_names import ModelSpec
from harness.progress import RolloutProgress, snapshot_output_size
from harness.prompt_log import log_tools

STREAM_STALL_TIMEOUT_SECONDS = 300
PREWARM_MAX_TOKENS = 32
PREWARM_USER_MESSAGE = "Prewarm the cached prompt prefix. Reply OK."


def _tool_defs(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


def _tool_map(tools: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}


def usage_from_response(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        dumped = usage.model_dump(exclude_none=True)
    else:
        dumped = {
            key: getattr(usage, key)
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
            if getattr(usage, key, None) is not None
        }
    details = dumped.get("output_tokens_details")
    if isinstance(details, dict) and details.get("thinking_tokens") is not None:
        dumped["thinking_tokens"] = details["thinking_tokens"]
    return dumped


def _usage_dict(response: Any) -> dict[str, Any]:
    return usage_from_response(response)


def _merge_usage(total: dict[str, Any], turn_usage: dict[str, Any]) -> None:
    for key, value in turn_usage.items():
        if isinstance(value, int):
            existing = total.get(key, 0)
            total[key] = existing + value if isinstance(existing, int) else value
        elif isinstance(value, dict):
            nested = total.get(key)
            if not isinstance(nested, dict):
                nested = {}
                total[key] = nested
            _merge_usage(nested, value)
        else:
            total[key] = value


def _assistant_block_to_dict(block: Any) -> dict[str, Any]:
    if hasattr(block, "model_dump"):
        dumped = block.model_dump(exclude_none=True)
        if dumped.get("type"):
            return dumped
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if block.type == "thinking":
        payload: dict[str, Any] = {"type": "thinking", "thinking": block.thinking}
        signature = getattr(block, "signature", None)
        if signature:
            payload["signature"] = signature
        return payload
    if block.type == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    return {"type": block.type}


def _message_from_transcript_turn(turn: dict[str, Any]) -> dict[str, Any] | None:
    role = turn.get("role")
    if role == "assistant":
        content = turn.get("content")
        if isinstance(content, list):
            return {"role": "assistant", "content": content}
        return None
    if role == "user":
        content = turn.get("content")
        if isinstance(content, list):
            return {"role": "user", "content": content}
        text = turn.get("text")
        if isinstance(text, str):
            return {"role": "user", "content": [{"type": "text", "text": text}]}
    return None


def _messages_from_transcript(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        message = _message_from_transcript_turn(turn)
        if message is not None:
            messages.append(message)
    return messages


def _strip_cache_control(content: Any) -> Any:
    if isinstance(content, list):
        return [_strip_cache_control(block) for block in content]
    if isinstance(content, dict):
        cleaned = {k: _strip_cache_control(v) for k, v in content.items() if k != "cache_control"}
        return cleaned
    return content


def _prewarm_request(*, spec: ModelSpec, system: str, cache_ttl: str) -> dict[str, Any]:
    """Build a cheap cache-write request: no thinking, tiny max_tokens."""
    return {
        "model": spec.model_id,
        "max_tokens": PREWARM_MAX_TOKENS,
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": _cache_control(cache_ttl),
            }
        ],
        "messages": [{"role": "user", "content": PREWARM_USER_MESSAGE}],
    }


def _cache_control(ttl: str) -> dict[str, str]:
    cache_control = {"type": "ephemeral"}
    if ttl == "1h":
        cache_control["ttl"] = "1h"
    return cache_control


def _apply_cache_control(
    messages: list[dict[str, Any]],
    *,
    cache_ttl: str,
    cache_first_user: bool,
) -> list[dict[str, Any]]:
    cleaned = _strip_cache_control(messages)
    if not cleaned:
        return cleaned

    first_user = cleaned[0]
    if cache_first_user and first_user.get("role") == "user" and isinstance(first_user.get("content"), list):
        content = list(first_user["content"])
        if content:
            content[-1] = {**content[-1], "cache_control": _cache_control(cache_ttl)}
            cleaned[0] = {**first_user, "content": content}

    for idx in range(len(cleaned) - 1, -1, -1):
        message = cleaned[idx]
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list) or not content:
            continue
        if content[0].get("type") != "tool_result":
            continue
        content = list(content)
        content[-1] = {**content[-1], "cache_control": _cache_control(cache_ttl)}
        cleaned[idx] = {**message, "content": content}
        break

    return cleaned


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        timeout = httpx.Timeout(connect=10.0, read=STREAM_STALL_TIMEOUT_SECONDS, write=30.0, pool=10.0)
        self._client = anthropic.Anthropic(
            api_key=api_key or get_api_key("anthropic"),
            timeout=timeout,
        )

    def prewarm_cache(self, *, spec: ModelSpec, system: str, cache_ttl: str) -> dict[str, int]:
        response = self._create(**_prewarm_request(spec=spec, system=system, cache_ttl=cache_ttl))
        return _usage_dict(response)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _create_non_streaming(self, **kwargs: Any) -> Any:
        return self._client.messages.create(**kwargs)

    def _create(
        self,
        *,
        progress: RolloutProgress | None = None,
        api_turn: int = 0,
        **kwargs: Any,
    ) -> Any:
        max_tokens = int(kwargs.get("max_tokens", 4096))
        thinking = kwargs.get("thinking")
        if thinking or max_tokens > 8192:
            if progress:
                progress.on_api_turn_start(api_turn)
            with self._client.messages.stream(**kwargs) as stream:
                last_progress = time.monotonic()
                last_size = 0
                streamed_chars = 0
                visible_chars = 0
                current_block_type = "starting"
                last_logged_chars = 0
                for event in stream:
                    now = time.monotonic()
                    if progress is None:
                        continue
                    if event.type == "message_start":
                        progress.on_message_start(_usage_dict(event.message))
                        last_progress = now
                    elif event.type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", "")
                        if delta_type == "thinking_delta":
                            streamed_chars += len(getattr(delta, "thinking", "") or "")
                            current_block_type = "thinking"
                            visible_chars = streamed_chars
                            progress.update_stream(block_type="thinking", output_chars=streamed_chars)
                            last_progress = now
                        elif delta_type == "text_delta":
                            streamed_chars += len(getattr(delta, "text", "") or "")
                            current_block_type = "text"
                            visible_chars = streamed_chars
                            progress.update_stream(block_type="text", output_chars=streamed_chars)
                            last_progress = now
                        elif delta_type == "input_json_delta":
                            streamed_chars += len(getattr(delta, "partial_json", "") or "")
                            current_block_type = "tool_use"
                            visible_chars = streamed_chars
                            progress.update_stream(block_type="tool_use", output_chars=streamed_chars)
                            last_progress = now
                    elif event.type == "thinking":
                        current_block_type = "thinking"
                        visible_chars = len(event.snapshot)
                        progress.update_stream(block_type="thinking", output_chars=visible_chars)
                        if len(event.snapshot) > last_size:
                            last_progress = now
                            last_size = len(event.snapshot)
                    elif event.type == "text":
                        current_block_type = "text"
                        visible_chars = len(event.snapshot)
                        progress.update_stream(block_type="text", output_chars=visible_chars)
                        if len(event.snapshot) > last_size:
                            last_progress = now
                            last_size = len(event.snapshot)
                    elif event.type == "input_json":
                        current_block_type = "tool_use"
                        progress.update_stream(block_type="tool_use")
                    elif event.type == "message_delta":
                        usage = getattr(event, "usage", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                        if output_tokens is not None:
                            progress.update_stream(output_tokens=int(output_tokens))
                            if int(output_tokens) > last_size:
                                last_progress = now
                                last_size = int(output_tokens)
                    if hasattr(stream, "current_message_snapshot"):
                        try:
                            size, block_type = snapshot_output_size(stream.current_message_snapshot)
                            if block_type:
                                current_block_type = block_type
                            visible_chars = max(visible_chars, size)
                            progress.update_stream(output_chars=visible_chars, block_type=current_block_type)
                            if size > last_size:
                                last_progress = now
                                last_size = size
                        except AssertionError:
                            pass
                    if now - last_progress > STREAM_STALL_TIMEOUT_SECONDS:
                        raise TimeoutError(
                            f"Anthropic stream stalled for {STREAM_STALL_TIMEOUT_SECONDS} seconds "
                            f"during API turn {api_turn}"
                        )
                    if progress and visible_chars - last_logged_chars >= 25_000:
                        last_logged_chars = visible_chars
                        progress.log(
                            f"api turn {api_turn} streaming "
                            f"({current_block_type}, {visible_chars:,} output chars so far)"
                        )
                return stream.get_final_message()

        if progress:
            progress.on_api_turn_start(api_turn)
        return self._create_non_streaming(**kwargs)

    def run_rollout(
        self,
        *,
        spec: ModelSpec,
        system: str,
        prompt: str,
        tools: list[Tool],
        max_turns: int,
        cache_ttl: str = "5m",
        max_token_continues: int = 0,
        resume_from: Transcript | None = None,
        progress: RolloutProgress | None = None,
    ) -> Transcript:
        tool_by_name = _tool_map(tools)

        if resume_from is None:
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
                },
                {"role": "user", "text": prompt},
            ]
            messages: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            usage: dict[str, int] = {}
            tool_call_count = 0
        else:
            transcript_turns = list(resume_from.turns)
            messages = _messages_from_transcript(transcript_turns)
            usage = dict(resume_from.usage)
            tool_call_count = resume_from.tool_call_count

        max_token_continue_count = sum(
            1
            for turn in transcript_turns
            if turn.get("role") == "user" and turn.get("reason") == "continue_after_max_tokens"
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
            request: dict[str, Any] = {
                "model": spec.model_id,
                "max_tokens": spec.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": _cache_control(cache_ttl),
                    }
                ],
                "messages": _apply_cache_control(
                    messages,
                    cache_ttl=cache_ttl,
                    cache_first_user=bool(tools),
                ),
            }
            if tools:
                request["tools"] = _tool_defs(tools)
            if spec.thinking:
                request["thinking"] = spec.thinking
            if spec.output_config:
                request["output_config"] = spec.output_config

            turn_started = time.monotonic()
            turn_started_at = datetime.now(timezone.utc).isoformat()
            response = self._create(**request, progress=progress, api_turn=api_turn)
            turn_duration_ms = int((time.monotonic() - turn_started) * 1000)
            turn_finished_at = datetime.now(timezone.utc).isoformat()
            turn_usage = _usage_dict(response)
            _merge_usage(usage, turn_usage)

            text_parts: list[str] = []
            tool_uses: list[dict[str, Any]] = []
            assistant_content: list[dict[str, Any]] = []

            for block in response.content:
                assistant_content.append(_assistant_block_to_dict(block))
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )

            assistant_text = "\n".join(text_parts) if text_parts else None
            transcript_turns.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "stop_reason": response.stop_reason,
                    "usage": turn_usage,
                    "started_at": turn_started_at,
                    "finished_at": turn_finished_at,
                    "duration_ms": turn_duration_ms,
                }
            )
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "max_tokens" and max_token_continue_count < max_token_continues:
                max_token_continue_count += 1
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "continue"}],
                    }
                )
                transcript_turns.append(
                    {
                        "role": "user",
                        "text": "continue",
                        "reason": "continue_after_max_tokens",
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

            if response.stop_reason == "tool_use" and tool_uses:
                tool_result_blocks: list[dict[str, Any]] = []
                for call in tool_uses:
                    tool_call_count += 1
                    tool = tool_by_name.get(call["name"])
                    if tool is None:
                        result = f"Unknown tool: {call['name']}"
                    else:
                        try:
                            result = tool.run(**call["input"])
                        except Exception as exc:  # noqa: BLE001
                            result = f"Tool error: {exc}"
                    tool_result_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call["id"],
                            "content": result,
                        }
                    )

                messages.append({"role": "user", "content": tool_result_blocks})
                transcript_turns.append(
                    {
                        "role": "user",
                        "content": tool_result_blocks,
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
