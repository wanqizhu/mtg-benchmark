from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.config import DEFAULT_CONCURRENCY, DEFAULT_JUDGE_MODEL, DEFAULT_MAX_TURNS, RESULTS_DIR, get_model
from harness.model_names import ModelSpec, safe_result_dir_name
from harness.core import Benchmark, Sample, Transcript
from harness.pricing import estimate_cost
from harness.progress import RolloutProgress
from harness.providers.anthropic import PROMPT_CACHE_TTL, AnthropicProvider
from harness.providers.base import Provider


def _result_path(run_dir: Path, model_name: str, sample_id: str) -> Path:
    return run_dir / safe_result_dir_name(model_name) / f"{sample_id}.json"


def _judge_path(run_dir: Path, model_name: str, sample_id: str) -> Path:
    return run_dir / safe_result_dir_name(model_name) / f"{sample_id}.judge.json"


def _progress_summary(transcript: Transcript) -> dict[str, Any]:
    assistant = [turn for turn in transcript.turns if turn.get("role") == "assistant"]
    continues = sum(
        1
        for turn in transcript.turns
        if turn.get("role") == "user" and turn.get("reason") == "continue_after_max_tokens"
    )
    last_stop = assistant[-1].get("stop_reason") if assistant else None
    return {
        "assistant_turns": len(assistant),
        "continues": continues,
        "last_stop_reason": last_stop,
        "tool_call_count": transcript.tool_call_count,
    }


def _transcript_to_dict(transcript: Transcript) -> dict[str, Any]:
    return asdict(transcript)


def _load_transcript(path: Path) -> Transcript:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript(**data)


def _error_payload(exc: BaseException) -> dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    payload: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if status_code is not None:
        payload["status_code"] = status_code
        payload["client_error"] = 400 <= int(status_code) < 500
    if body is not None:
        payload["body"] = body
    return payload


def get_provider(spec: ModelSpec) -> Provider:
    if spec.provider == "anthropic":
        return AnthropicProvider()
    raise NotImplementedError(f"Provider not implemented: {spec.provider}")


class Runner:
    def __init__(
        self,
        benchmark: Benchmark,
        *,
        run_id: str | None = None,
        results_dir: Path = RESULTS_DIR,
        max_turns: int = DEFAULT_MAX_TURNS,
        concurrency: int = DEFAULT_CONCURRENCY,
        rules_mode: str = "tools",
        max_tokens: int | None = None,
        max_token_continues: int = 0,
        judge_model: str | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.results_dir = results_dir
        self.run_dir = results_dir / self.run_id
        self.max_turns = max_turns
        self.concurrency = concurrency
        self.rules_mode = rules_mode
        self.max_tokens = max_tokens
        self.max_token_continues = max_token_continues
        self.judge_model = judge_model or getattr(benchmark, "judge_model", None) or DEFAULT_JUDGE_MODEL

    def _samples_for(self, sample_ids: list[str] | None) -> list[Sample]:
        samples = self.benchmark.samples()
        if not sample_ids:
            return samples
        wanted = set(sample_ids)
        selected = [sample for sample in samples if sample.id in wanted]
        missing = wanted - {sample.id for sample in selected}
        if missing:
            raise KeyError(f"Unknown sample ids: {', '.join(sorted(missing))}")
        return selected

    async def run_rollouts(
        self,
        model_names: list[str],
        *,
        sample_ids: list[str] | None = None,
        force: bool = False,
        resume: bool = False,
    ) -> list[Path]:
        samples = self._samples_for(sample_ids)
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: list[asyncio.Task[Path | None]] = []

        for model_name in model_names:
            spec = get_model(model_name)
            if self.max_tokens is not None:
                spec = replace(spec, max_tokens=self.max_tokens)
            provider = get_provider(spec)
            pending = [
                (sample, _result_path(self.run_dir, model_name, sample.id))
                for sample in samples
                if (
                    _result_path(self.run_dir, model_name, sample.id).exists()
                    if resume
                    else force or not _result_path(self.run_dir, model_name, sample.id).exists()
                )
            ]
            if pending and self.rules_mode == "inline":
                prewarm_cache = getattr(provider, "prewarm_cache", None)
                if prewarm_cache is not None:
                    sample, _ = pending[0]
                    usage = await asyncio.to_thread(
                        prewarm_cache,
                        spec=spec,
                        system=sample.system,
                    )
                    print(
                        f"[{model_name}] prewarmed cache "
                        f"(input={usage.get('input_tokens', 0):,}, "
                        f"output={usage.get('output_tokens', 0):,}, "
                        f"cache_read={usage.get('cache_read_input_tokens', 0):,}, "
                        f"cache_create={usage.get('cache_creation_input_tokens', 0):,})",
                        file=sys.stderr,
                        flush=True,
                    )
                    if usage:
                        prewarm_payload = {
                            "run_id": self.run_id,
                            "model_name": model_name,
                            "model_id": spec.model_id,
                            "cache_ttl": PROMPT_CACHE_TTL,
                            "thinking": spec.thinking,
                            "output_config": spec.output_config,
                            "usage": usage,
                            "cost": estimate_cost(
                                usage,
                                model_id=spec.model_id,
                                cache_ttl=PROMPT_CACHE_TTL,
                            ),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        self.run_dir.mkdir(parents=True, exist_ok=True)
                        prewarm_path = self.run_dir / f"prewarm.{model_name}.json"
                        prewarm_path.write_text(
                            json.dumps(prewarm_payload, indent=2) + "\n",
                            encoding="utf-8",
                        )

            for sample, out_path in pending:
                tasks.append(
                    asyncio.create_task(
                        self._run_one(provider, spec, model_name, sample, out_path, semaphore)
                        if not resume
                        else self._resume_one(provider, spec, model_name, sample, out_path, semaphore)
                    )
                )

        results = await asyncio.gather(*tasks)
        return [path for path in results if path is not None]

    async def _run_one(
        self,
        provider: Provider,
        spec: ModelSpec,
        model_name: str,
        sample: Sample,
        out_path: Path,
        semaphore: asyncio.Semaphore,
    ) -> Path:
        async with semaphore:
            validate = getattr(self.benchmark, "validate_model", None)
            if validate is not None:
                validate(spec)

            max_turns = 1 if not sample.tools else self.max_turns

            def payload_for(transcript: Transcript, status: str) -> dict[str, Any]:
                payload = {
                    "benchmark": self.benchmark.name,
                    "run_id": self.run_id,
                    "status": status,
                    "rules_mode": self.rules_mode,
                    "cache_ttl": PROMPT_CACHE_TTL,
                    "model_name": model_name,
                    "model_id": spec.model_id,
                    "sample_id": sample.id,
                    "model_config": {
                        "thinking": spec.thinking,
                        "output_config": spec.output_config,
                        "max_tokens": spec.max_tokens,
                        "max_turns": max_turns,
                        "max_token_continues": self.max_token_continues,
                    },
                    "progress": _progress_summary(transcript),
                }
                if transcript.usage:
                    payload["cost"] = estimate_cost(
                        transcript.usage,
                        model_id=spec.model_id,
                        cache_ttl=PROMPT_CACHE_TTL,
                    )
                payload["reference"] = sample.reference
                payload["transcript"] = _transcript_to_dict(transcript)
                return payload

            progress = RolloutProgress(
                model_name=model_name,
                sample_id=sample.id,
                out_path=out_path,
                payload_for=payload_for,
                run_id=self.run_id,
                model_id=spec.model_id,
                cache_ttl=PROMPT_CACHE_TTL,
                live_path=self.results_dir / "live.jsonl",
            )
            try:
                transcript = await asyncio.to_thread(
                    provider.run_rollout,
                    spec=spec,
                    system=sample.system,
                    prompt=sample.prompt,
                    tools=sample.tools,
                    max_turns=max_turns,
                    max_token_continues=self.max_token_continues,
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001
                if out_path.exists():
                    existing = json.loads(out_path.read_text(encoding="utf-8"))
                    transcript = Transcript(**existing.get("transcript", {}))
                else:
                    transcript = Transcript(model=spec.model_id, provider=provider.provider_name)
                progress.on_rollout_failed(transcript)
                payload = payload_for(transcript, "failed")
                payload["error"] = _error_payload(exc)
                payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(
                    f"[{model_name} {sample.id}] rollout failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return out_path
            if out_path.exists():
                return out_path

            payload = payload_for(transcript, "complete")
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return out_path

    async def _resume_one(
        self,
        provider: Provider,
        spec: ModelSpec,
        model_name: str,
        sample: Sample,
        out_path: Path,
        semaphore: asyncio.Semaphore,
    ) -> Path:
        async with semaphore:
            validate = getattr(self.benchmark, "validate_model", None)
            if validate is not None:
                validate(spec)

            max_turns = 1 if not sample.tools else self.max_turns
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            transcript = Transcript(**existing["transcript"])

            def payload_for(next_transcript: Transcript, status: str) -> dict[str, Any]:
                payload = {
                    "benchmark": existing.get("benchmark", self.benchmark.name),
                    "run_id": existing.get("run_id", self.run_id),
                    "status": status,
                    "rules_mode": existing.get("rules_mode", self.rules_mode),
                    "cache_ttl": existing.get("cache_ttl", PROMPT_CACHE_TTL),
                    "model_name": model_name,
                    "model_id": spec.model_id,
                    "sample_id": sample.id,
                    "model_config": {
                        "thinking": spec.thinking,
                        "output_config": spec.output_config,
                        "max_tokens": spec.max_tokens,
                        "max_turns": max_turns,
                        "max_token_continues": self.max_token_continues,
                    },
                    "progress": _progress_summary(next_transcript),
                }
                if next_transcript.usage:
                    payload["cost"] = estimate_cost(
                        next_transcript.usage,
                        model_id=spec.model_id,
                        cache_ttl=PROMPT_CACHE_TTL,
                    )
                payload["reference"] = existing.get("reference", sample.reference)
                payload["transcript"] = _transcript_to_dict(next_transcript)
                return payload

            progress = RolloutProgress(
                model_name=model_name,
                sample_id=sample.id,
                out_path=out_path,
                payload_for=payload_for,
                run_id=self.run_id,
                model_id=spec.model_id,
                cache_ttl=PROMPT_CACHE_TTL,
                live_path=self.results_dir / "live.jsonl",
            )
            try:
                resumed = await asyncio.to_thread(
                    provider.run_rollout,
                    spec=spec,
                    system=sample.system,
                    prompt=sample.prompt,
                    tools=sample.tools,
                    max_turns=max_turns,
                    max_token_continues=self.max_token_continues,
                    resume_from=transcript,
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001
                current = json.loads(out_path.read_text(encoding="utf-8"))
                current["status"] = "failed"
                current["error"] = _error_payload(exc)
                current["updated_at"] = datetime.now(timezone.utc).isoformat()
                out_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
                print(
                    f"[{model_name} {sample.id}] resume failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return out_path

            payload = payload_for(resumed, "complete")
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return out_path

    async def run_judges(
        self,
        model_names: list[str],
        *,
        sample_ids: list[str] | None = None,
        force: bool = False,
    ) -> list[Path]:
        samples = {sample.id: sample for sample in self._samples_for(sample_ids)}
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks: list[asyncio.Task[Path | None]] = []

        for model_name in model_names:
            for sample_id, sample in samples.items():
                result_path = _result_path(self.run_dir, model_name, sample_id)
                judge_path = _judge_path(self.run_dir, model_name, sample_id)
                if not result_path.exists():
                    continue
                if judge_path.exists() and not force:
                    continue
                result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                if result_payload.get("status") != "complete":
                    continue
                tasks.append(
                    asyncio.create_task(
                        self._judge_one(sample, result_path, judge_path, semaphore)
                    )
                )

        results = await asyncio.gather(*tasks)
        return [path for path in results if path is not None]

    async def _judge_one(
        self,
        sample: Sample,
        result_path: Path,
        judge_path: Path,
        semaphore: asyncio.Semaphore,
    ) -> Path:
        async with semaphore:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            transcript = Transcript(**payload["transcript"])
            verdict = await asyncio.to_thread(self.benchmark.judge, sample, transcript)
            judge_spec = get_model(self.judge_model)
            judge_payload = {
                "benchmark": self.benchmark.name,
                "run_id": self.run_id,
                "sample_id": sample.id,
                "model_name": payload["model_name"],
                "judge_model_name": judge_spec.name,
                "judge_model_id": judge_spec.model_id,
                "verdict": asdict(verdict),
            }
            if verdict.usage:
                judge_payload["usage"] = verdict.usage
                judge_payload["cost"] = estimate_cost(
                    verdict.usage,
                    model_id=judge_spec.model_id,
                )
            judge_path.parent.mkdir(parents=True, exist_ok=True)
            judge_path.write_text(json.dumps(judge_payload, indent=2), encoding="utf-8")
            return judge_path
