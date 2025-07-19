# tests/ci/test_reflex_log_score.py
# 🧪 Unit tests for src/ci/reflex_log_score.py

import os
import tempfile
from src.ci import reflex_log_score

def test_score_reflex_log_text_all_markers_present():
    log_text = (
        "Pressure mutated: True\n"
        "Pressure solver invoked: True\n"
        "Projection attempted: True\n"
        "Projection skipped: True\n"
        "[🔄 Step 0 Summary]\n"
    )
    result = reflex_log_score.score_reflex_log_text(log_text)
    assert result["reflex_score"] == "5 / 5"
    assert set(result["markers_matched"]) == {
        "pressure_mutation",
        "projection_triggered",
        "projection_attempted",
        "projection_skipped",
        "step_summary_detected"
    }

def test_score_reflex_log_text_partial_markers():
    log_text = "Pressure mutated: True\nProjection attempted: True\n"
    result = reflex_log_score.score_reflex_log_text(log_text)
    assert result["reflex_score"] == "2 / 5"
    assert set(result["markers_matched"]) == {"pressure_mutation", "projection_attempted"}

def test_score_reflex_log_text_no_markers():
    log_text = "Simulation initialized.\nNo diagnostic markers found.\n"
    result = reflex_log_score.score_reflex_log_text(log_text)
    assert result["reflex_score"] == "0 / 5"
    assert result["markers_matched"] == []

def test_score_from_summary_file_mock(monkeypatch):
    # Simulate evaluate_reflex_score return structure
    def mock_evaluate(path):
        return {"steps_scored": 3, "average_score": 0.6}

    monkeypatch.setattr(reflex_log_score, "evaluate_reflex_score", mock_evaluate)

    # Create a dummy file path
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("dummy summary")
        summary_path = f.name

    result = reflex_log_score.score_from_summary_file(summary_path)
    os.unlink(summary_path)
    assert result == {"steps_scored": 3, "average_score": 0.6}

def test_score_combined_output_structure(monkeypatch):
    def mock_evaluate(path):
        return {"steps_scored": 10, "average_score": 0.7}
    monkeypatch.setattr(reflex_log_score, "evaluate_reflex_score", mock_evaluate)

    log_text = "Pressure mutated: True\nProjection attempted: True\n"
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("dummy")
        summary_path = f.name

    result = reflex_log_score.score_combined(log_text, summary_path)
    os.unlink(summary_path)

    assert result["ci_log_score"]["reflex_score"] == "2 / 5"
    assert "pressure_mutation" in result["ci_log_score"]["markers_matched"]
    assert result["summary_score"]["steps_scored"] == 10
    assert result["summary_score"]["average_score"] == 0.7



