# tests/ci/test_reflex_log_score.py

import pytest
from ci import reflex_log_score

@pytest.fixture
def sample_log_text():
    return (
        "Pressure updated @\n"
        "Mutation pathway recorded →\n"
        "Pressure delta map saved →"
    )

@pytest.fixture
def summary_file(tmp_path):
    content = """[🔄 Step 0 Summary]
• Pressure mutated: True
• Divergence max: 3.92e-3
• Suppression count: 2
"""
    file_path = tmp_path / "step_summary.txt"
    file_path.write_text(content)
    return str(file_path)

def test_score_reflex_log_text_full_match():
    log_text = "\n".join(reflex_log_score.MARKERS.keys())
    result = reflex_log_score.score_reflex_log_text(log_text)
    assert result["reflex_score"] == "5 / 5"
    assert set(result["markers_matched"]) == set(reflex_log_score.MARKERS.values())

def test_score_reflex_log_text_partial(sample_log_text):
    result = reflex_log_score.score_reflex_log_text(sample_log_text)
    assert result["reflex_score"] == "3 / 5"
    assert set(result["markers_matched"]) == {
        "pressure_mutation",
        "mutation_pathway",
        "pressure_delta"
    }

def test_score_from_summary_file(summary_file):
    result = reflex_log_score.score_from_summary_file(summary_file)
    assert isinstance(result, dict)
    assert "step_scores" in result
    assert 0 in result["step_scores"]

def test_score_combined(sample_log_text, summary_file):
    result = reflex_log_score.score_combined(sample_log_text, summary_file)
    assert "ci_log_score" in result
    assert "summary_score" in result
    assert result["ci_log_score"]["reflex_score"].startswith("3 /")



