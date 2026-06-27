import pytest

from elice_rag.eval.judge import parse_judge_response


def test_parse_judge_response_from_fenced_json():
    result = parse_judge_response(
        """```json
        {
          "groundedness": 1,
          "correctness": 0.5,
          "judge_pass": "false",
          "rationale": "부분적으로만 기준을 만족합니다."
        }
        ```"""
    )

    assert result.groundedness == 1.0
    assert result.correctness == 0.5
    assert result.judge_score == 0.75
    assert result.judge_pass is False
    assert result.rationale == "부분적으로만 기준을 만족합니다."


def test_parse_judge_response_rejects_non_json():
    with pytest.raises(ValueError):
        parse_judge_response("groundedness is high")
