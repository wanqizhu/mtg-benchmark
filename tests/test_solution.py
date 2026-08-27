from unittest.mock import MagicMock

from benchmarks.mtg.bench import MTGBenchmark
from benchmarks.mtg.solution import extract_model_answer
from harness.core import Sample, Transcript


def test_extract_model_answer_requires_solution_tags():
    transcript = Transcript(
        turns=[
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Thinking..."},
                    {"type": "text", "text": "<solution>\n1. Attack.\n</solution>"},
                ],
            }
        ]
    )
    assert extract_model_answer(transcript) == "1. Attack."


def test_extract_model_answer_from_top_level_text_field():
    transcript = Transcript(
        turns=[
            {
                "role": "assistant",
                "text": "<solution>\n1. Attack.\n</solution>",
                "thinking": "Thinking...",
            }
        ]
    )
    assert extract_model_answer(transcript) == "1. Attack."


def test_extract_model_answer_missing_tags():
    transcript = Transcript(
        turns=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "1. Attack for lethal."}],
            }
        ]
    )
    assert extract_model_answer(transcript) is None


def test_extract_model_answer_empty_transcript():
    assert extract_model_answer(Transcript()) is None


def test_judge_skips_api_without_final_answer():
    benchmark = MTGBenchmark()
    sample = Sample(
        id="001",
        system="",
        prompt="",
        tools=[],
        reference={"problem_gold_md": "puzzle", "solution_text": "1. Win."},
    )
    transcript = Transcript(
        turns=[
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "No tagged answer."}],
            }
        ]
    )
    benchmark._judge = MagicMock()

    verdict = benchmark.judge(sample, transcript)

    benchmark._judge.judge.assert_not_called()
    assert verdict.passed is False
    assert verdict.reasoning == "No final answer provided."