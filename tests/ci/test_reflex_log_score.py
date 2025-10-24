# tests/ci/test_reflex_log_score.py
# ✅ Validation suite for src/ci/reflex_log_score.py

import os
import pytest
from src.ci.reflex_log_score import (
    score_reflex_log_text,
    score_from_summary_file,
    score_combined,
    MARKERS
)

@pytest.mark.parametrize("log_text,expected_score,expected_labels", [
    (
        "\n".join(MARKERS.keys()),
        f"{len(MARKERS)} / {len(MARKERS)}",
        list(MARKERS.values())
    ),
    (
        "Pressure mutated: True\nProjection attempted: True",
        "2 / 5",
        ["pressure_mutation", "projection_attempted"]
    ),
    (
        "No relevant markers here",
        "0 / 5",
        []
    ),
])
def test_score_reflex_log_text_variants(log_text, expected_score, expected_labels, capsys):
    result = score_reflex_log_text(log_text)
    assert result["reflex_score"] == expected_score
    assert set(result["markers_matched"]) == set(expected_labels)
    assert "[CI SCORE] Matched" in capsys.readouterr().out

def test_score_from_summary_file_valid(tmp_path, monkeypatch, capsys):
    summary_path = tmp_path / "step_summary.txt"
    summary_path.write_text("Step 0001\nPressure mutated: True\nProjection attempted: True\n")

    monkeypatch.setattr(
        "src.ci.reflex_log_score.evaluate_reflex_score",
        lambda path: {
            "step_scores": [{"step": 1, "score": "2 / 5"}],
            "summary": {"average_score": "2 / 5"}
        }
    )

    result = score_from_summary_file(str(summary_path))
    assert "step_scores" in result
    assert "summary" in result
    assert result["summary"]["average_score"] == "2 / 5"
    assert "[CI SCORE] Summary file evaluated" in capsys.readouterr().out

def test_score_from_summary_file_missing(tmp_path, monkeypatch, capsys):
    missing_path = tmp_path / "missing_summary.txt"

    monkeypatch.setattr(
        "src.ci.reflex_log_score.evaluate_reflex_score",
        lambda path: {"error": "File not found"} if not os.path.exists(path) else {}
    )

    result = score_from_summary_file(str(missing_path))
    assert "error" in result
    assert "[CI SCORE] Summary file evaluated" in capsys.readouterr().out

def test_score_combined_valid(tmp_path, monkeypatch, capsys):
    log_text = "Pressure mutated: True\nProjection skipped: True"
    summary_path = tmp_path / "step_summary.txt"
    summary_path.write_text("Step 0002\nPressure mutated: True\nProjection skipped: True\n")

    monkeypatch.setattr(
        "src.ci.reflex_log_score.evaluate_reflex_score",
        lambda path: {
            "step_scores": [{"step": 2, "score": "2 / 5"}],
            "summary": {"average_score": "2 / 5"}
        }
    )

    result = score_combined(log_text, str(summary_path))
    assert "ci_log_score" in result
    assert "summary_score" in result
    assert result["ci_log_score"]["reflex_score"] == "2 / 5"
    assert result["summary_score"]["summary"]["average_score"] == "2 / 5"
    assert "[CI SCORE] Combined scoring complete." in capsys.readouterr().out



