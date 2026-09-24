from __future__ import annotations

from typing import Literal

from harness.config import DEFAULT_JUDGE_MODEL, get_model
from harness.core import Sample, Transcript, Verdict
from harness.judge import Judge, make_judge
from harness.model_names import ModelSpec
from harness.tools import FileTools

from benchmarks.mtg.dataset import clarifications_path, load_puzzles, load_rules_text, rules_path
from benchmarks.mtg.prompts import (
    JUDGE_SYSTEM,
    judge_user_prompt,
    solver_system_prompt_inline,
    solver_system_prompt_tools,
    solver_user_prompt,
)
from benchmarks.mtg.solution import extract_model_answer, parse_verdict

RulesMode = Literal["tools", "inline"]

# Rules alone are ~232k tokens; require headroom for puzzle, thinking, and output.
INLINE_MIN_CONTEXT_TOKENS = 250_000

MODEL_CONTEXT_TOKENS: dict[str, int] = {
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-opus-5-5": 1_000_000,
    "claude-opus-5": 1_000_000,
    "claude-opus-4-8": 1_000_000,
    "claude-opus-4-7": 1_000_000,
    "claude-opus-4-6": 1_000_000,
    "claude-opus-4-5-20251101": 200_000,
    "claude-fable-5-1": 1_000_000,
    "claude-fable-5": 1_000_000,
    "gpt-5.6-sol": 1_050_000,
    "gpt-5.6-terra": 1_050_000,
    "gpt-5.6-luna": 1_050_000,
    "gpt-6-sol": 1_050_000,
    "gpt-6-luna": 1_050_000,
    "gpt-6-astra": 1_050_000,
    "grok-4.7": 500_000,
    "grok-4.6": 500_000,
    "grok-4.5": 500_000,
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-3.8-flash": 1_048_576,
    "gemini-3.7-flash": 1_048_576,
    "gemini-3.6-flash": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-3.1-flash-lite": 1_048_576,
    # OpenRouter ids vary per model; unknown ids fall back to 200k (tools mode only).
}


class MTGBenchmark:
    name = "mtg"

    def __init__(self, *, rules_mode: RulesMode = "tools", judge_model: str | None = None) -> None:
        if rules_mode not in {"tools", "inline"}:
            raise ValueError(f"Unknown rules_mode: {rules_mode!r}")
        self.rules_mode = rules_mode
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL
        self._rules_path = rules_path()
        self._clarifications = clarifications_path().read_text(encoding="utf-8")
        self._rules_text = load_rules_text() if rules_mode == "inline" else ""
        self._file_tools = FileTools([self._rules_path])
        self._judge: Judge | None = None

    def validate_model(self, spec: ModelSpec) -> None:
        if self.rules_mode != "inline":
            return
        context = MODEL_CONTEXT_TOKENS.get(spec.model_id, 200_000)
        if context < INLINE_MIN_CONTEXT_TOKENS:
            raise ValueError(
                f"rules_mode=inline requires a model with at least "
                f"{INLINE_MIN_CONTEXT_TOKENS:,} tokens of context; "
                f"{spec.name} ({spec.model_id}) has {context:,}. "
                f"Use Sonnet 5 / 4.6+, Opus 4.6+, Fable 5.1 / 5, "
                f"GPT-5.6 Sol, or Grok 4.5+."
            )

    def samples(self) -> list[Sample]:
        if self.rules_mode == "tools":
            system = solver_system_prompt_tools(self._clarifications, str(self._rules_path))
            tools = self._file_tools.as_tools()
            inline = False
        else:
            system = solver_system_prompt_inline(self._clarifications, self._rules_text)
            tools = []
            inline = True

        samples: list[Sample] = []
        for puzzle in load_puzzles():
            samples.append(
                Sample(
                    id=puzzle.id,
                    system=system,
                    prompt=solver_user_prompt(puzzle.gold_md, inline=inline),
                    tools=tools,
                    reference={
                        "difficulty": puzzle.difficulty,
                        "solution_text": puzzle.solution_text,
                        "problem_gold_md": puzzle.gold_md,
                        "rules_path": str(self._rules_path),
                        "rules_mode": self.rules_mode,
                    },
                )
            )
        return samples

    def _get_judge(self) -> Judge:
        if self._judge is None:
            self._judge = make_judge(get_model(self.judge_model))
        return self._judge

    def judge(self, sample: Sample, transcript: Transcript) -> Verdict:
        model_answer = extract_model_answer(transcript)
        if not model_answer:
            return Verdict(
                passed=False,
                reasoning="No final answer provided.",
                raw={"verdict": "fail", "reasoning": "No final answer provided."},
            )

        user_prompt = judge_user_prompt(
            problem_text=sample.reference.get("problem_gold_md", ""),
            model_answer=model_answer,
            official_solution=sample.reference.get("solution_text", ""),
        )
        response_text, usage = self._get_judge().judge(
            system=JUDGE_SYSTEM,
            user=user_prompt,
        )
        raw = parse_verdict(response_text)
        verdict = raw["verdict"]
        passed = verdict in {"pass", "pass_with_unique_solution"}
        reasoning = raw["reasoning"]
        return Verdict(
            passed=passed,
            reasoning=reasoning,
            raw=raw,
            prompt={"system": JUDGE_SYSTEM, "user": user_prompt},
            usage=usage,
        )
