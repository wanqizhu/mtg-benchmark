from types import SimpleNamespace

import pytest

from harness.judge import final_text


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
