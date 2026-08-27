import pytest

from benchmarks.mtg.solution import parse_verdict


def test_parse_verdict_from_tags():
    text = """After review:

<verdict>
{"verdict": "pass", "reasoning": "Legal winning line."}
</verdict>
"""
    result = parse_verdict(text)
    assert result["verdict"] == "pass"
    assert result["reasoning"] == "Legal winning line."


def test_parse_verdict_reasoning_with_mana_symbols():
    text = """<verdict>
{"verdict": "fail", "reasoning": "Cannot pay {2}{R} with available mana."}
</verdict>"""
    result = parse_verdict(text)
    assert result["verdict"] == "fail"
    assert "{2}{R}" in result["reasoning"]


def test_parse_verdict_pass_with_unique_solution():
    text = """<verdict>
{"verdict": "pass_with_unique_solution", "reasoning": "Different but valid."}
</verdict>"""
    assert parse_verdict(text)["verdict"] == "pass_with_unique_solution"


def test_parse_verdict_missing_tags():
    with pytest.raises(ValueError, match="No <verdict> block"):
        parse_verdict("No structured response here.")
