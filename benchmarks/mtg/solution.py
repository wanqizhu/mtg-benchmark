from __future__ import annotations

import json
import re

from harness.core import Transcript


def parse_solution(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip() or None


def parse_verdict(text: str) -> dict[str, str]:
    match = re.search(r"<verdict>(.*?)</verdict>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("No <verdict> block found in judge response")
    payload = match.group(1).strip()
    raw = json.loads(payload)
    return {
        "verdict": str(raw.get("verdict", "fail")).lower(),
        "reasoning": str(raw.get("reasoning", "")),
    }


def last_assistant_text(transcript: Transcript) -> str | None:
    for turn in reversed(transcript.turns):
        if turn.get("role") != "assistant":
            continue
        text = turn.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        content = turn.get("content")
        if isinstance(content, list):
            joined = "\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text" and block.get("text")
            ).strip()
            if joined:
                return joined
    return None


def extract_model_answer(transcript: Transcript) -> str | None:
    text = last_assistant_text(transcript)
    if not text:
        return None
    return parse_solution(text)
