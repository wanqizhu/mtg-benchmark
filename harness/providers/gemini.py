"""Gemini provider over the native google-genai SDK.

Gemini 3 requires `thought_signature` to be echoed back on function-call parts,
or the next request fails with 400. The signatures ride inside the raw model
`Content` parts, so this provider stores each model turn as `api_content`
(the serialized Content) and replays exactly those parts on the next request.
https://ai.google.dev/gemini-api/docs/thought-signatures
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from google import genai
from google.genai import types

from harness.config import DEFAULT_MAX_TOKEN_CONTINUES, get_api_key
from harness.core import Tool, Transcript
from harness.model_names import ModelSpec
from harness.providers.base import append_max_token_continue
from harness.progress import RolloutProgress
from harness.prompt_log import log_tools

REQUEST_TIMEOUT_SECONDS = 3600
CACHE_TTL = "auto"
STOP_REASON_MAP = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
}


def _tool_defs(tools: list[Tool]) -> list[types.Tool]:
    if not tools:
        return []
    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters_json_schema=tool.input_schema,
                )
                for tool in tools
            ]
        )
    ]


def usage_from_gemini(usage: Any) -> dict[str, Any]:
    """Normalize usage_metadata onto Anthropic-shaped fields.

    Gemini bills thinking at the output rate; candidates_token_count excludes
    thoughts, so output_tokens = candidates + thoughts. Cached prompt tokens are
    a subset of prompt_token_count.
    """
    if usage is None:
        return {}
    raw = usage if isinstance(usage, dict) else usage.model_dump(exclude_none=True)
    prompt = int(raw.get("prompt_token_count") or 0)
    candidates = int(raw.get("candidates_token_count") or 0)
    thoughts = int(raw.get("thoughts_token_count") or 0)
    cached = int(raw.get("cached_content_token_count") or 0)
    # Tool-use prompt tokens (function declarations) are billed as input.
    tool_prompt = int(raw.get("tool_use_prompt_token_count") or 0)
    normalized: dict[str, Any] = {
        "input_tokens": max(0, prompt + tool_prompt - cached),
        "output_tokens": candidates + thoughts,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
        "prompt_tokens": prompt + tool_prompt,
        "completion_tokens": candidates,
    }
    if thoughts:
        normalized["thinking_tokens"] = thoughts
    return normalized


def _merge_usage(total: dict[str, Any], turn_usage: dict[str, Any]) -> None:
    for key, value in turn_usage.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            existing = total.get(key, 0)
            total[key] = existing + value if isinstance(existing, (int, float)) else value
        else:
            total[key] = value


def content_to_dict(content: types.Content) -> dict[str, Any]:
    return content.model_dump(exclude_none=True, mode="json")


def content_from_dict(payload: dict[str, Any]) -> types.Content:
    return types.Content.model_validate(payload)


def _call_id(call: types.FunctionCall, index: int) -> str:
    return call.id or f"{call.name}-{index}"


def content_blocks(content: types.Content) -> list[dict[str, Any]]:
    """Convert model Content parts to the harness transcript block shape."""
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []
    for index, part in enumerate(content.parts or []):
        if part.function_call is not None:
            tool_blocks.append(
                {
                    "type": "tool_use",
                    "id": _call_id(part.function_call, index),
                    "name": part.function_call.name,
                    "input": dict(part.function_call.args or {}),
                }
            )
        elif part.text:
            (thinking_parts if part.thought else text_parts).append(part.text)
    blocks: list[dict[str, Any]] = []
    if thinking_parts:
        blocks.append({"type": "thinking", "thinking": "".join(thinking_parts)})
    if text_parts:
        blocks.append({"type": "text", "text": "".join(text_parts)})
    blocks.extend(tool_blocks)
    return blocks


def map_stop_reason(finish_reason: Any, blocks: list[dict[str, Any]]) -> str | None:
    if any(block.get("type") == "tool_use" for block in blocks):
        return "tool_use"
    if finish_reason is None:
        return None
    name = getattr(finish_reason, "name", None) or str(finish_reason)
    return STOP_REASON_MAP.get(name, name.lower())


def merge_stream_parts(parts: list[types.Part]) -> list[types.Part]:
    """Join streamed text deltas into one part per run, keeping signatures.

    Consecutive text parts with the same `thought` flag are concatenated. A
    thought_signature stays on the part it arrived with; if the accumulating
    part already carries one, a new part is started so no signature is lost.
    Function-call and other parts pass through untouched.
    """
    merged: list[types.Part] = []
    for part in parts:
        is_text = part.text is not None and part.function_call is None and part.function_response is None
        if not is_text:
            merged.append(part)
            continue
        if not part.text and part.thought_signature is None:
            continue
        last = merged[-1] if merged else None
        can_join = (
            last is not None
            and last.text is not None
            and last.function_call is None
            and bool(last.thought) == bool(part.thought)
            and not (last.thought_signature is not None and part.thought_signature is not None)
        )
        if can_join:
            last.text = (last.text or "") + (part.text or "")
            if part.thought_signature is not None:
                last.thought_signature = part.thought_signature
        else:
            merged.append(part.model_copy())
    return merged


def contents_from_transcript(turns: list[dict[str, Any]]) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in turns:
        api_content = turn.get("api_content")
        if isinstance(api_content, dict):
            contents.append(content_from_dict(api_content))
            continue
        if turn.get("role") == "user" and isinstance(turn.get("text"), str):
            contents.append(types.Content(role="user", parts=[types.Part(text=turn["text"])]))
    return contents


class GeminiProvider:
    provider_name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        self._client = genai.Client(
            api_key=api_key or get_api_key("gemini"),
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    http_status_codes=[429, 500, 502, 503, 504],
                ),
            ),
        )

    def _config(self, spec: ModelSpec, system: str, tools: list[Tool]) -> types.GenerateContentConfig:
        thinking = spec.thinking or {}
        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=spec.max_tokens,
            tools=_tool_defs(tools) or None,
            thinking_config=types.ThinkingConfig(
                thinking_level=thinking.get("thinking_level", "high"),
                include_thoughts=bool(thinking.get("include_thoughts", True)),
            ),
            # The harness runs tools itself; never let the SDK auto-call them.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _create(
        self,
        *,
        progress: RolloutProgress | None,
        api_turn: int,
        spec: ModelSpec,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> tuple[types.Content, dict[str, Any], Any]:
        if progress:
            progress.on_api_turn_start(api_turn)

        parts: list[types.Part] = []
        finish_reason: Any = None
        usage: dict[str, Any] = {}
        announced_usage = False
        visible_chars = 0
        last_logged_chars = 0

        stream = self._client.models.generate_content_stream(
            model=spec.model_id,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            if chunk.usage_metadata is not None:
                usage = usage_from_gemini(chunk.usage_metadata)
                if progress and not announced_usage and usage.get("prompt_tokens"):
                    announced_usage = True
                    progress.on_message_start(usage)
            if chunk.prompt_feedback is not None and chunk.prompt_feedback.block_reason:
                raise RuntimeError(
                    f"Gemini blocked the prompt: {chunk.prompt_feedback.block_reason} "
                    f"{chunk.prompt_feedback.block_reason_message or ''}".strip()
                )
            candidate = chunk.candidates[0] if chunk.candidates else None
            if candidate is None:
                continue
            if candidate.finish_reason is not None:
                finish_reason = candidate.finish_reason
            if candidate.content is None or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                parts.append(part)
                if part.function_call is not None:
                    block = "tool_use"
                    visible_chars += len(json.dumps(part.function_call.args or {}))
                elif part.text:
                    block = "thinking" if part.thought else "text"
                    visible_chars += len(part.text)
                else:
                    continue
                if progress:
                    progress.update_stream(block_type=block, output_chars=visible_chars)
                    if visible_chars - last_logged_chars >= 25_000:
                        last_logged_chars = visible_chars
                        progress.log(
                            f"api turn {api_turn} streaming ({block}, {visible_chars:,} output chars so far)"
                        )

        if progress:
            progress.update_stream(output_tokens=int(usage.get("output_tokens") or 0))
        content = types.Content(role="model", parts=merge_stream_parts(parts))
        return content, usage, finish_reason

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
        tool_by_name = {tool.name: tool for tool in tools}
        config = self._config(spec, system, tools)

        if resume_from is None:
            first = types.Content(role="user", parts=[types.Part(text=prompt)])
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
                    "cache_ttl": CACHE_TTL,
                    "api": "generate_content",
                },
                {"role": "user", "text": prompt, "api_content": content_to_dict(first)},
            ]
            contents: list[types.Content] = [first]
            usage: dict[str, Any] = {}
            tool_call_count = 0
        else:
            transcript_turns = list(resume_from.turns)
            usage = dict(resume_from.usage)
            tool_call_count = resume_from.tool_call_count

        max_token_continue_count = sum(
            1
            for turn in transcript_turns
            if turn.get("role") == "user" and turn.get("reason") == "continue_after_max_tokens"
        )
        if resume_from is not None:
            queued = max_token_continue_count
            cont = types.Content(role="user", parts=[types.Part(text="continue")])
            max_token_continue_count = append_max_token_continue(
                transcript_turns,
                max_token_continue_count,
                max_token_continues,
                api_content=content_to_dict(cont),
            )
            contents = contents_from_transcript(transcript_turns)
            if max_token_continue_count != queued and progress:
                progress.log(
                    f"continuing after max_tokens "
                    f"({max_token_continue_count}/{max_token_continues})"
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
        while sum(1 for turn in transcript_turns if turn.get("role") == "assistant") < max_api_turns:
            api_turn = sum(1 for turn in transcript_turns if turn.get("role") == "assistant") + 1
            turn_started = time.monotonic()
            turn_started_at = datetime.now(timezone.utc).isoformat()
            content, turn_usage, finish_reason = self._create(
                progress=progress,
                api_turn=api_turn,
                spec=spec,
                contents=contents,
                config=config,
            )
            turn_duration_ms = int((time.monotonic() - turn_started) * 1000)
            turn_finished_at = datetime.now(timezone.utc).isoformat()
            _merge_usage(usage, turn_usage)
            blocks = content_blocks(content)
            stop_reason = map_stop_reason(finish_reason, blocks)
            text_parts = [block["text"] for block in blocks if block.get("type") == "text"]
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
                    "api_content": content_to_dict(content),
                }
            )
            contents.append(content)

            if stop_reason == "max_tokens" and max_token_continue_count < max_token_continues:
                max_token_continue_count += 1
                cont = types.Content(role="user", parts=[types.Part(text="continue")])
                contents.append(cont)
                transcript_turns.append(
                    {
                        "role": "user",
                        "text": "continue",
                        "reason": "continue_after_max_tokens",
                        "api_content": content_to_dict(cont),
                    }
                )
                if progress:
                    progress.on_turn_finished(current_transcript())
                    progress.log(
                        f"continuing after max_tokens "
                        f"({max_token_continue_count}/{max_token_continues})"
                    )
                continue

            calls = [
                (index, part.function_call)
                for index, part in enumerate(content.parts or [])
                if part.function_call is not None
            ]
            if calls:
                tool_result_blocks: list[dict[str, Any]] = []
                response_parts: list[types.Part] = []
                for index, call in calls:
                    tool_call_count += 1
                    tool = tool_by_name.get(call.name or "")
                    arguments = dict(call.args or {})
                    try:
                        if tool is None:
                            result = f"Unknown tool: {call.name}"
                        else:
                            result = tool.run(**arguments)
                    except Exception as exc:  # noqa: BLE001
                        result = f"Tool error: {exc}"
                    call_id = _call_id(call, index)
                    tool_result_blocks.append(
                        {"type": "tool_result", "tool_use_id": call_id, "content": result}
                    )
                    response_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=call.id,
                                name=call.name,
                                response={"result": result},
                            )
                        )
                    )
                # All function responses go in one user Content, in call order
                # (interleaving with the calls is a 400 on Gemini 3).
                reply = types.Content(role="user", parts=response_parts)
                contents.append(reply)
                transcript_turns.append(
                    {
                        "role": "user",
                        "content": tool_result_blocks,
                        "api_content": content_to_dict(reply),
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
