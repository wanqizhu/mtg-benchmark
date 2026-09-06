from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from harness.config import get_api_key
from harness.core import Tool, Transcript
from harness.model_names import ModelSpec
from harness.progress import RolloutProgress
from harness.prompt_log import log_tools

STREAM_STALL_TIMEOUT_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 3600
CACHE_TTL = "auto"


def _dump(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    raise TypeError(f"Expected dict-like API object, got {type(value).__name__}")


def _tool_defs(tools: list[Tool]) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []
    for tool in tools:
        parameters = dict(tool.input_schema)
        parameters.setdefault("additionalProperties", False)
        definitions.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
                "strict": True,
            }
        )
    return definitions


def usage_from_openai(usage: Any) -> dict[str, Any]:
    """Normalize Responses API usage without double-counting reasoning.

    OpenAI's output_tokens already includes output_tokens_details.reasoning_tokens.
    """
    if not usage:
        return {}
    raw = _dump(usage)
    input_tokens = int(raw.get("input_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    cached = int(input_details.get("cached_tokens") or 0)
    cache_write = int(input_details.get("cache_write_tokens") or 0)
    reasoning = int(output_details.get("reasoning_tokens") or 0)
    normalized: dict[str, Any] = {
        "input_tokens": max(0, input_tokens - cached - cache_write),
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cached,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "input_tokens_details": input_details,
        "output_tokens_details": output_details,
    }
    if cache_write:
        normalized["cache_write_input_tokens"] = cache_write
    if reasoning:
        normalized["thinking_tokens"] = reasoning
    return normalized


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


def response_content_blocks(response: Any) -> list[dict[str, Any]]:
    """Convert typed Responses output items to the harness transcript shape."""
    raw = _dump(response)
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []

    for item_value in raw.get("output") or []:
        item = _dump(item_value)
        item_type = item.get("type")
        if item_type == "reasoning":
            for summary_value in item.get("summary") or []:
                summary = _dump(summary_value)
                if summary.get("text"):
                    thinking_parts.append(summary["text"])
        elif item_type == "message":
            for content_value in item.get("content") or []:
                content = _dump(content_value)
                if content.get("type") == "output_text" and content.get("text"):
                    text_parts.append(content["text"])
                elif content.get("type") == "refusal" and content.get("refusal"):
                    text_parts.append(content["refusal"])
        elif item_type == "function_call":
            arguments = item.get("arguments") or "{}"
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = {"_raw": arguments}
            tool_blocks.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": parsed,
                    "response_item_id": item.get("id"),
                }
            )

    blocks: list[dict[str, Any]] = []
    if thinking_parts:
        blocks.append({"type": "thinking", "thinking": "\n\n".join(thinking_parts)})
    if text_parts:
        blocks.append({"type": "text", "text": "\n".join(text_parts)})
    blocks.extend(tool_blocks)
    return blocks


def map_stop_reason(response: Any, blocks: list[dict[str, Any]]) -> str:
    raw = _dump(response)
    if any(block.get("type") == "tool_use" for block in blocks):
        return "tool_use"
    if raw.get("status") == "incomplete":
        details = raw.get("incomplete_details") or {}
        if details.get("reason") == "max_output_tokens":
            return "max_tokens"
    if raw.get("status") == "completed":
        return "end_turn"
    return str(raw.get("status") or "unknown")


def _resume_input(turns: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    previous_response_id: str | None = None
    last_assistant_index = -1
    for index, turn in enumerate(turns):
        if turn.get("role") == "assistant" and turn.get("response_id"):
            previous_response_id = turn["response_id"]
            last_assistant_index = index
    if previous_response_id:
        for turn in turns[last_assistant_index + 1 :]:
            api_input = turn.get("api_input")
            if isinstance(api_input, list):
                return previous_response_id, api_input
        return previous_response_id, [{"role": "user", "content": "continue"}]

    for turn in reversed(turns):
        api_input = turn.get("api_input")
        if isinstance(api_input, list):
            return None, api_input
    return None, []


class OpenAIProvider:
    provider_name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        self._client = OpenAI(
            api_key=api_key or get_api_key("openai"),
            timeout=float(REQUEST_TIMEOUT_SECONDS),
            max_retries=2,
        )

    def _create(
        self,
        *,
        progress: RolloutProgress | None,
        api_turn: int,
        request: dict[str, Any],
    ) -> Any:
        if progress:
            progress.on_api_turn_start(api_turn)
        visible_chars = 0
        last_progress = time.monotonic()
        last_logged_chars = 0

        with self._client.responses.stream(**request) as stream:
            for event in stream:
                now = time.monotonic()
                event_type = event.type
                delta = getattr(event, "delta", "") or ""
                if event_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    visible_chars += len(delta)
                    last_progress = now
                    if progress:
                        progress.update_stream(
                            block_type="thinking", output_chars=visible_chars
                        )
                elif event_type == "response.output_text.delta":
                    visible_chars += len(delta)
                    last_progress = now
                    if progress:
                        progress.update_stream(block_type="text", output_chars=visible_chars)
                elif event_type == "response.function_call_arguments.delta":
                    visible_chars += len(delta)
                    last_progress = now
                    if progress:
                        progress.update_stream(
                            block_type="tool_use", output_chars=visible_chars
                        )
                elif event_type == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise RuntimeError(f"OpenAI response failed: {error}")

                if now - last_progress > STREAM_STALL_TIMEOUT_SECONDS:
                    raise TimeoutError(
                        f"OpenAI stream stalled for {STREAM_STALL_TIMEOUT_SECONDS} seconds "
                        f"during API turn {api_turn}"
                    )
                if progress and visible_chars - last_logged_chars >= 25_000:
                    last_logged_chars = visible_chars
                    progress.log(
                        f"api turn {api_turn} streaming "
                        f"({visible_chars:,} output chars so far)"
                    )
            response = stream.get_final_response()

        turn_usage = usage_from_openai(response.usage)
        if progress:
            progress.on_message_start(turn_usage)
            progress.update_stream(
                output_tokens=int(turn_usage.get("output_tokens") or 0)
            )
        return response

    def run_rollout(
        self,
        *,
        spec: ModelSpec,
        system: str,
        prompt: str,
        tools: list[Tool],
        max_turns: int,
        max_token_continues: int = 0,
        resume_from: Transcript | None = None,
        progress: RolloutProgress | None = None,
    ) -> Transcript:
        tool_by_name = {tool.name: tool for tool in tools}
        reasoning = {
            "effort": str((spec.output_config or {}).get("effort") or "medium"),
            **(spec.thinking or {}),
        }
        tool_defs = _tool_defs(tools)

        if resume_from is None:
            prompt_cache_key = str(uuid.uuid4())
            first_input = [{"role": "user", "content": prompt}]
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
                    "prompt_cache_key": prompt_cache_key,
                    "cache_ttl": CACHE_TTL,
                    "api": "responses",
                },
                {"role": "user", "text": prompt, "api_input": first_input},
            ]
            previous_response_id: str | None = None
            next_input = first_input
            usage: dict[str, Any] = {}
            tool_call_count = 0
        else:
            transcript_turns = list(resume_from.turns)
            config = next(
                (turn for turn in transcript_turns if turn.get("role") == "config"), {}
            )
            prompt_cache_key = config.get("prompt_cache_key") or str(uuid.uuid4())
            previous_response_id, next_input = _resume_input(transcript_turns)
            usage = dict(resume_from.usage)
            tool_call_count = resume_from.tool_call_count

        max_token_continue_count = sum(
            1
            for turn in transcript_turns
            if turn.get("role") == "user"
            and turn.get("reason") == "continue_after_max_tokens"
        )

        def current_transcript() -> Transcript:
            return Transcript(
                turns=transcript_turns,
                tool_call_count=tool_call_count,
                usage=usage,
                model=spec.model_id,
                provider=self.provider_name,
            )

        if progress:
            progress.write(current_transcript(), status="in_progress")
            progress.log("rollout resumed" if resume_from is not None else "rollout started")

        max_api_turns = max_turns + max_token_continues
        while sum(
            1 for turn in transcript_turns if turn.get("role") == "assistant"
        ) < max_api_turns:
            api_turn = (
                sum(1 for turn in transcript_turns if turn.get("role") == "assistant")
                + 1
            )
            request: dict[str, Any] = {
                "model": spec.model_id,
                "instructions": system,
                "input": next_input,
                "reasoning": reasoning,
                "max_output_tokens": spec.max_tokens,
                "prompt_cache_key": prompt_cache_key,
                "store": True,
            }
            if previous_response_id:
                request["previous_response_id"] = previous_response_id
            if tool_defs:
                request["tools"] = tool_defs

            turn_started = time.monotonic()
            turn_started_at = datetime.now(timezone.utc).isoformat()
            response = self._create(
                progress=progress,
                api_turn=api_turn,
                request=request,
            )
            turn_duration_ms = int((time.monotonic() - turn_started) * 1000)
            turn_finished_at = datetime.now(timezone.utc).isoformat()
            turn_usage = usage_from_openai(response.usage)
            _merge_usage(usage, turn_usage)
            blocks = response_content_blocks(response)
            stop_reason = map_stop_reason(response, blocks)
            raw_response = _dump(response)
            text_parts = [
                block["text"] for block in blocks if block.get("type") == "text"
            ]
            transcript_turns.append(
                {
                    "role": "assistant",
                    "content": blocks,
                    "text": "\n".join(text_parts) if text_parts else None,
                    "stop_reason": stop_reason,
                    "usage": turn_usage,
                    "started_at": turn_started_at,
                    "finished_at": turn_finished_at,
                    "duration_ms": turn_duration_ms,
                    "response_id": response.id,
                    "response_status": response.status,
                    "api_output": raw_response.get("output") or [],
                }
            )
            previous_response_id = response.id

            if stop_reason == "max_tokens" and max_token_continue_count < max_token_continues:
                max_token_continue_count += 1
                next_input = [{"role": "user", "content": "continue"}]
                transcript_turns.append(
                    {
                        "role": "user",
                        "text": "continue",
                        "reason": "continue_after_max_tokens",
                        "api_input": next_input,
                    }
                )
                if progress:
                    progress.on_turn_finished(current_transcript())
                    progress.log(
                        f"continuing after max_tokens "
                        f"({max_token_continue_count}/{max_token_continues})"
                    )
                continue

            tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
            if tool_uses:
                tool_results: list[dict[str, Any]] = []
                next_input = []
                for call in tool_uses:
                    tool_call_count += 1
                    name = call.get("name")
                    tool = tool_by_name.get(name)
                    arguments = call.get("input")
                    try:
                        if not isinstance(arguments, dict) or "_raw" in arguments:
                            raise ValueError(f"Invalid tool arguments: {arguments!r}")
                        if tool is None:
                            result = f"Unknown tool: {name}"
                        else:
                            result = tool.run(**arguments)
                    except Exception as exc:  # noqa: BLE001
                        result = f"Tool error: {exc}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.get("id"),
                            "content": result,
                        }
                    )
                    next_input.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.get("id"),
                            "output": result,
                        }
                    )
                transcript_turns.append(
                    {
                        "role": "user",
                        "content": tool_results,
                        "api_input": next_input,
                    }
                )
                if progress:
                    progress.on_turn_finished(current_transcript())
                continue

            if progress:
                progress.on_turn_finished(current_transcript())
            break

        final = current_transcript()
        if progress:
            progress.on_rollout_complete(final)
        return final
