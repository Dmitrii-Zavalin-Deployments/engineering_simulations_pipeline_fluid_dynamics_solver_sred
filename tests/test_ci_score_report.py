import os
import pytest
import builtins
from unittest import mock

import src.ci_score_report as ci_report

@pytest.fixture
def mock_summary_file(tmp_path):
    summary_path = tmp_path / "step_summary.txt"
    content = """
    [🔄 Step 001 Summary]
    • Reflex score: 0.42
    • Mutated cells: 3
    • Projection attempted: True
    """
    summary_path.write_text(content)
    return summary_path

def test_missing_summary_file(monkeypatch, capsys):
    monkeypatch.setattr(ci_report, "SUMMARY_PATH", "nonexistent/path/step_summary.txt")
    with mock.patch("builtins.print") as mock_print:
        ci_report.__name__ = "__main__"
        exec(open("src/ci_score_report.py").read(), {"__name__": "__main__"})
        mock_print.assert_any_call("❌ Summary file not found → nonexistent/path/step_summary.txt")

def test_score_combined_invoked(monkeypatch, mock_summary_file):
    dummy_scores = {
        "ci_log_score": {"markers_matched": 5, "reflex_score": 0.42},
        "summary_score": {"average_score": 0.42}
    }

    monkeypatch.setattr(ci_report, "SUMMARY_PATH", str(mock_summary_file))
    monkeypatch.setattr(ci_report, "score_combined", lambda text, path: dummy_scores)

    with mock.patch("builtins.print") as mock_print:
        ci_report.__name__ = "__main__"
        exec(open("src/ci_score_report.py").read(), {"__name__": "__main__"})
        mock_print.assert_any_call("📊 CI Reflex Scoring Results:")
        mock_print.assert_any_call("↪️ Matched markers: 5")
        mock_print.assert_any_call("↪️ Marker score: 0.42")
        mock_print.assert_any_call("↪️ Summary score: 0.42 (avg)")
        mock_print.assert_any_call("⚠️ Reflex score below expected threshold.")



