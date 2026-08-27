from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AccessDeniedError(PermissionError):
    pass


@dataclass
class FileTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    _fn: Any

    def run(self, **kwargs: Any) -> str:
        return self._fn(**kwargs)


class FileTools:
    """Generic read-only file tools with an injected access policy."""

    def __init__(self, allowed_paths: list[Path]) -> None:
        self._allowed = {p.resolve() for p in allowed_paths}
        self._cache: dict[Path, list[str]] = {}
        for path in allowed_paths:
            resolved = path.resolve()
            self._cache[resolved] = resolved.read_text(encoding="utf-8").splitlines()

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            raise AccessDeniedError(f"Path must be absolute: {path}")
        resolved = candidate.resolve()
        if resolved not in self._allowed:
            raise AccessDeniedError(f"Access denied: {path}")
        return resolved

    def _lines(self, path: str) -> list[str]:
        resolved = self._resolve(path)
        if resolved not in self._cache:
            self._cache[resolved] = resolved.read_text(encoding="utf-8").splitlines()
        return self._cache[resolved]

    def grep(self, path: str, pattern: str) -> str:
        lines = self._lines(path)
        regex = re.compile(pattern, re.IGNORECASE)
        matches: list[str] = []
        for idx, line in enumerate(lines, start=1):
            if regex.search(line):
                matches.append(f"{idx}:{line}")
        if not matches:
            return "No matches found."
        if len(matches) > 200:
            head = matches[:200]
            return "\n".join(head) + f"\n... truncated, {len(matches) - 200} more matches"
        return "\n".join(matches)

    def read(self, path: str, line_number: int, num_lines: int) -> str:
        if line_number < 1:
            raise ValueError("line_number must be >= 1")
        if num_lines < 1:
            raise ValueError("num_lines must be >= 1")
        lines = self._lines(path)
        start = line_number - 1
        if start >= len(lines):
            return "No lines at that offset."
        end = min(start + num_lines, len(lines))
        selected = lines[start:end]
        return "\n".join(f"{start + i + 1}:{selected[i]}" for i in range(len(selected)))

    def as_tools(self) -> list[FileTool]:
        allowed_display = ", ".join(str(p) for p in sorted(self._allowed))
        return [
            FileTool(
                name="grep",
                description=(
                    "Search a file for lines matching a regex pattern. "
                    f"Allowed paths: {allowed_display}"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to search.",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for.",
                        },
                    },
                    "required": ["path", "pattern"],
                },
                _fn=self.grep,
            ),
            FileTool(
                name="read",
                description=(
                    "Read a window of lines from a file, each prefixed with its line number. "
                    f"Allowed paths: {allowed_display}"
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the file to read.",
                        },
                        "line_number": {
                            "type": "integer",
                            "description": "1-based line number to start reading from.",
                        },
                        "num_lines": {
                            "type": "integer",
                            "description": "Number of lines to return.",
                        },
                    },
                    "required": ["path", "line_number", "num_lines"],
                },
                _fn=self.read,
            ),
        ]
