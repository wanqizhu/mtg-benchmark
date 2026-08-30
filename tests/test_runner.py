import json

from harness.runner import complete_result_ready_for_judge


def test_complete_result_ready_for_judge(tmp_path):
    result = tmp_path / "001.json"
    judge = tmp_path / "001.judge.json"
    assert complete_result_ready_for_judge(result, judge) is False

    result.write_text(json.dumps({"status": "in_progress"}), encoding="utf-8")
    assert complete_result_ready_for_judge(result, judge) is False

    result.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    assert complete_result_ready_for_judge(result, judge) is True

    judge.write_text("{}", encoding="utf-8")
    assert complete_result_ready_for_judge(result, judge) is False
    assert complete_result_ready_for_judge(result, judge, force=True) is True