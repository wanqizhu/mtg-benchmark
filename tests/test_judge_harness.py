from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness.judge import final_text, final_text_from_openai, make_judge
from harness.model_names import parse_model_name


def test_final_text_returns_last_text_block():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", text="work through the line"),
            SimpleNamespace(type="text", text="draft verdict"),
            SimpleNamespace(type="text", text="final verdict"),
        ]
    )
    assert final_text(response) == "final verdict"


def test_final_text_single_text_block():
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", text="thinking only before answer"),
            SimpleNamespace(type="text", text="only text"),
        ]
    )
    assert final_text(response) == "only text"


def test_final_text_missing_text_block():
    response = SimpleNamespace(content=[SimpleNamespace(type="thinking", text="hmm")])
    with pytest.raises(ValueError, match="No text block"):
        final_text(response)


def test_final_text_from_openai_uses_last_output_text():
    response = {
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "grade the line"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "draft"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "final"}],
            },
        ]
    }
    assert final_text_from_openai(response) == "draft\nfinal"


def test_make_judge_dispatches_by_provider():
    with patch("harness.judge.OpenAI"), patch("harness.judge.get_api_key", return_value="sk-test"):
        openai_judge = make_judge(parse_model_name("gpt-5.6-sol-thinking-high"))
    with patch("harness.judge.anthropic.Anthropic"), patch(
        "harness.judge.get_api_key", return_value="sk-ant"
    ):
        anthropic_judge = make_judge(parse_model_name("claude-sonnet-5-thinking-high"))
    assert type(openai_judge).__name__ == "OpenAIJudge"
    assert type(anthropic_judge).__name__ == "AnthropicJudge"


def test_make_judge_rejects_unsupported_provider():
    spec = parse_model_name("grok-4.6-high")
    with pytest.raises(ValueError, match="No judge implementation"):
        make_judge(spec)
