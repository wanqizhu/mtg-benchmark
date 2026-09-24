from __future__ import annotations

from typing import Any, Protocol

from harness.config import DEFAULT_MAX_TOKEN_CONTINUES
from harness.model_names import ModelSpec
from harness.core import Tool, Transcript
from harness.progress import RolloutProgress


def append_max_token_continue(
    turns: list[dict[str, Any]],
    count: int,
    limit: int,
    **fields: Any,
) -> int:
    """Record the user turn that follows a saved max_tokens stop.

    A resumed rollout already stopped on the cap, so this turn has to be queued
    before the next request. Returns the updated continue count.
    """
    if count >= limit or not turns:
        return count
    last = turns[-1]
    if last.get("role") != "assistant" or last.get("stop_reason") != "max_tokens":
        return count
    turns.append(
        {
            "role": "user",
            "text": "continue",
            "reason": "continue_after_max_tokens",
            **fields,
        }
    )
    return count + 1


class Provider(Protocol):
    provider_name: str

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
    ) -> Transcript: ...
