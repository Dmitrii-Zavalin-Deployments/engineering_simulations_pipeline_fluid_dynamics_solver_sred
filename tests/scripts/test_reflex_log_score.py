# tests/ci/test_reflex_log_score.py

import pytest
from ci import reflex_log_score

@pytest.fixture
def complete_log_text():
    return "\n".join(reflex_log_score.MARKERS.keys())

@pytest.fixture
def partial_log_text():
    return "Pressure updated @\nPressure delta map saved →"

@pytest.fixture
def dummy_summary(tmp_path):
    path = tmp_path / "step_summary.txt"
    path.write_text("[🔄 Step 0 Summary]\n• Pressure mutated: True\n")
    return str(path)

def test_score_reflex_log_text_all_markers(complete_log_text):
    result = reflex_log_score.score_reflex_log_text(complete_log_text)
    assert result["reflex_score"] == "5 / 5"
    assert sorted(result["markers_matched"]) == sorted(reflex_log_score.MARKERS.values())

def test_score_reflex_log_text_some_markers(partial_log_text):
    result = reflex_log_score.score_reflex_log_text(partial_log_text)
    assert result["reflex_score"] == "2 / 5"
    assert set(result["markers_matched"]) == {"pressure_mutation", "pressure_delta"}

def test_score_reflex_log_text_no_match():
    result = reflex_log_score.score_reflex_log_text("some unrelated log content")
    assert result["reflex_score"] == "0 / 5"
    assert result["markers_matched"] == []

def test_score_from_summary_file(dummy_summary):
    result = reflex_log_score.score_from_summary_file(dummy_summary)
    assert isinstance(result, dict)
    assert "step_scores" in result
    assert isinstance(result["step_scores"], dict)
    assert 0 in result["step_scores"]

def test_score_combined_valid_inputs(complete_log_text, dummy_summary):
    result = reflex_log_score.score_combined(complete_log_text, dummy_summary)
    assert "ci_log_score" in result
    assert "summary_score" in result
    assert result["ci_log_score"]["reflex_score"] == "5 / 5"



