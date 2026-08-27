from __future__ import annotations

import re

_HEADER = re.compile(
    r"(?:^|\n)\s*(?:"
    r"Puzzle\s*#\d+\s*Solution\s*:?|"
    r"The Solution|"
    r"Here's the solution[^.\n]*|"
    r"Here is the solution[^.\n]*|"
    r"Looking for the solution to last week[^\n]*|"
    r"(?:Actual |Not quite there )?Solution(?:\s*(?:[\(-][^\n]*|[A-Z]\b))?"
    r")\s*:?\s*\n",
    re.I,
)

_FALLBACK = re.compile(
    r"(?:so here it is:|onto the solution\.?|get to the solution\.?|get down to the good stuff:)\s*\n",
    re.I,
)

_FOOTER = re.compile(
    r"^\s*(?:"
    r"Remember,|"
    r"View the refreshed|"
    r"We hope you had fun|"
    r"Thanks for playing|"
    r"Good luck!?|"
    r"Thanks again|"
    r"Have a great weekend|"
    r"\( \*\* Note:"
    r")",
    re.I,
)

_NUMBERED_STEP = re.compile(r"^\s*(\d+[a-z]?)(?:\s*\([^)]+\))?[\.)]\s*(.+)$", re.I)
_STEP_LABEL = re.compile(r"^\s*Step\s+(\d+)\s*:\s*(.+)$", re.I)
_BULLET_STEP = re.compile(r"^\s*[⦁•●]\s*(.+)$")

_GAMEPLAY_START = re.compile(
    r"^(?:"
    r"Cast|Play|Attack|Tap|Sacrifice|Activate|Use|Equip|Crew|Block|Pass|Target|Deal|Draw|Exile|"
    r"Return|Destroy|Copy|Transform|Go to|From |Pay |Hold |Trigger|\+|\-|Tutor|Discard|Mill|"
    r"Put |Create|Search|Reveal|Counter|Fight|Bounce|Blink|Untap|End |During|Before|After|"
    r"When |If |Note:|Repeat|Recast|Flashback|Overload|Escalate|Convoke|Delve"
    r")",
    re.I,
)

_POLLUTED_FIRST_STEP = (
    "i'm sending this out",
    "remember that your solution",
    "we should clarify",
    "yes, a vehicle can crew",
    "finally, please remember",
)


_BRANCH = re.compile(r"^\s*Branch\s", re.I)
_CONTINUATION = re.compile(r"^\s*-\s+")


def _dedupe_steps(steps: list[str]) -> list[str]:
    n = len(steps)
    if n >= 2 and n % 2 == 0 and steps[: n // 2] == steps[n // 2 :]:
        return steps[: n // 2]

    for period in range(1, n // 2 + 1):
        block = steps[:period]
        if len(block) >= 3 and steps[period : period + len(block)] == block:
            return block

    for index in range(1, len(steps)):
        if steps[index].startswith("1. "):
            prefix = steps[:index]
            if len(prefix) >= 3:
                return prefix

    return steps


def _match_step(line: str) -> tuple[str, str] | None:
    for pattern in (_NUMBERED_STEP, _STEP_LABEL):
        match = pattern.match(line)
        if match:
            return match.group(1), match.group(2).strip()
    bullet = _BULLET_STEP.match(line)
    if bullet:
        return "•", bullet.group(1).strip()
    return None


def _steps_from_body(body: str) -> list[str]:
    steps: list[str] = []
    started = False
    bullet_index = 0

    for line in body.splitlines():
        step_match = _match_step(line)
        if step_match:
            label, content = step_match
            if label == "•":
                bullet_index += 1
                steps.append(f"{bullet_index}. {content}")
            else:
                steps.append(f"{label}. {content}")
            started = True
            continue

        if started and _FOOTER.match(line):
            break

        stripped = line.strip()
        if started and stripped:
            if _BRANCH.match(stripped):
                steps.append(stripped)
                continue
            if _CONTINUATION.match(stripped):
                steps.append(stripped)
                continue

    return steps


def _is_polluted(steps: list[str]) -> bool:
    if not steps:
        return True
    first = steps[0].lower()
    return any(token in first for token in _POLLUTED_FIRST_STEP)


def _prose_from_body(body: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    for line in body.splitlines():
        if _FOOTER.match(line) and paragraphs:
            break
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if _match_step(stripped):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            break
        current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))

    gameplay = [p for p in paragraphs if _GAMEPLAY_START.match(p)]
    if not gameplay:
        return []
    return [f"{index}. {paragraph}" for index, paragraph in enumerate(gameplay, start=1)]


def _candidate_bodies(text: str) -> list[str]:
    matches = list(_HEADER.finditer(text))
    if matches:
        return [text[match.end() :] for match in matches]

    fallback = _FALLBACK.search(text)
    if fallback:
        return [text[fallback.end() :]]

    return [text]


def parse_official_solution(text: str) -> str:
    if not text.strip():
        return ""

    best_numbered: list[str] = []
    best_prose: list[str] = []

    for body in _candidate_bodies(text):
        numbered = _steps_from_body(body)
        if numbered and not _is_polluted(numbered) and len(numbered) > len(best_numbered):
            best_numbered = numbered
        elif numbered and not best_numbered and len(numbered) > len(best_numbered):
            best_numbered = numbered

        prose = _prose_from_body(body)
        if len(prose) > len(best_prose):
            best_prose = prose

    if best_numbered:
        return "\n".join(_dedupe_steps(best_numbered))
    if best_prose:
        return "\n".join(best_prose)
    return ""
