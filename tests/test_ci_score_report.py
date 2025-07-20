# ✅ Unit Test Suite — CI Score Report
# 📄 Full Path: tests/test_ci_score_report.py

import pytest
import os
from tempfile import TemporaryDirectory
from src.ci.reflex_log_score import score_combined

def test_score_combined_returns_expected_keys():
    log_text = """
    [🔄 Step 1 Summary]
    • Ghosts: 2
    • Fluid–ghost adjacents: 1
    • Influence applied: 2
    • Max divergence: 0.02
    • Projection attempted: True
    • Projection skipped: False
    • Pressure mutated: True
    • Reflex score: 0.65
    • Mutated cells: 3
    • Adaptive timestep: 0.01

    [🔄 Step 2 Summary]
    • Ghosts: 1
    • Fluid–ghost adjacents: 0
    • Influence applied: 1
    • Max divergence: 0.04
    • Projection attempted: True
    • Projection skipped: False
    • Pressure mutated: False
    • Reflex score: 0.55
    • Mutated cells: 1
    • Adaptive timestep: 0.02
    """
    with TemporaryDirectory() as tmp:
        summary_path = os.path.join(tmp, "step_summary.txt")
        with open(summary_path, "w") as f:
            f.write(log_text)

        scores = score_combined(log_text, summary_path)

        assert "ci_log_score" in scores
        assert "summary_score" in scores
        assert isinstance(scores["ci_log_score"]["markers_matched"], int)
        assert isinstance(scores["ci_log_score"]["reflex_score"], float)
        assert isinstance(scores["summary_score"]["average_score"], float)

def test_score_combined_handles_missing_fields():
    log_text = """
    [🔄 Step 3 Summary]
    • Reflex score: n/a
    • Pressure mutated: unknown
    """
    with TemporaryDirectory() as tmp:
        summary_path = os.path.join(tmp, "step_summary_missing.txt")
        with open(summary_path, "w") as f:
            f.write(log_text)

        scores = score_combined(log_text, summary_path)

        assert scores["summary_score"]["average_score"] >= 0.0
        assert scores["ci_log_score"]["markers_matched"] == 0

def test_score_combined_threshold_warning_triggered(capsys):
    log_text = """
    [🔄 Step 4 Summary]
    • Reflex score: 0.10
    • Pressure mutated: False
    • Adaptive timestep: 0.005
    """
    with TemporaryDirectory() as tmp:
        summary_path = os.path.join(tmp, "step_summary_low_score.txt")
        with open(summary_path, "w") as f:
            f.write(log_text)

        scores = score_combined(log_text, summary_path)
        assert scores["summary_score"]["average_score"] < 0.5



