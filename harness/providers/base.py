from __future__ import annotations

from typing import Protocol

from harness.model_names import ModelSpec
from harness.core import Tool, Transcript
from harness.progress import RolloutProgress


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
        max_token_continues: int = 0,
        resume_from: Transcript | None = None,
        progress: RolloutProgress | None = None,
    ) -> Transcript: ...
