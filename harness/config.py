from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from harness.model_names import ModelSpec, parse_model_name

load_dotenv()

DEFAULT_JUDGE_MODEL = "gpt-5.6-sol-thinking-high"
DEFAULT_MAX_TURNS = 20
# A turn that stops on the output cap was still thinking. Ask it to continue
# rather than grading the truncated turn. Five matches the Sonnet runs.
DEFAULT_MAX_TOKEN_CONTINUES = 5
DEFAULT_CONCURRENCY = 4
RESULTS_DIR = Path("results")


def get_model(name: str) -> ModelSpec:
    return parse_model_name(name)


def get_api_key(provider: str) -> str:
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "xai": "XAI_API_KEY",
    }
    env_name = env_map.get(provider)
    if not env_name:
        raise KeyError(f"No API key mapping for provider '{provider}'")
    value = os.getenv(env_name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable: {env_name}")
    return value
