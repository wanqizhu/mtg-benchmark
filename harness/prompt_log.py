from __future__ import annotations

from typing import Any

from harness.core import Tool


def log_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]
