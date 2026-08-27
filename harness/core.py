from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Transcript:
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    provider: str = ""


@dataclass
class Verdict:
    passed: bool
    reasoning: str
    raw: dict[str, Any] | None = None
    prompt: dict[str, str] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class Sample:
    id: str
    system: str
    prompt: str
    tools: list[Any]
    reference: Any


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def run(self, **kwargs: Any) -> str: ...


class Benchmark(Protocol):
    name: str

    def samples(self) -> list[Sample]: ...

    def judge(self, sample: Sample, transcript: Transcript) -> Verdict: ...
