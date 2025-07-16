# tests/metrics/test_reflex_score_evaluator.py

import os
import pytest
from src.metrics.reflex_score_evaluator import evaluate_reflex_score

TEST_DIR = "test_output/reflex_score_evaluator"

@pytest.fixture(autouse=True)
def setup_test_dir():
    os.makedirs(TEST_DIR, exist_ok=True)
    yield
    for fname in os.listdir(TEST_DIR):
        os.remove(os.path.join(TEST_DIR, fname))
    os.rmdir(TEST_DIR)

def write_summary(filename, content):
    with open(filename, "w") as f:
        f.write(content)

def test_evaluate_valid_summary():
    summary_path = os.path.join(TEST_DIR, "valid_summary.txt")
    write_summary(summary_path, """
[🔄 Step 0 Summary]
• Influence applied: 5
• Fluid–ghost adjacents: 3
• Pressure mutated: True

[🔄 Step 1 Summary]
• Influence applied: 10
• Fluid–ghost adjacents: 8
• Pressure mutated: False
""")
    report = evaluate_reflex_score(summary_path)
    assert report["step_count"] == 2
    assert report["max_score"] <= 1.0
    assert 0.0 < report["average_score"] < 1.0
    assert sorted(report["step_scores"].keys()) == [0, 1]

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        evaluate_reflex_score("nonexistent_path.txt")

def test_empty_summary_file():
    empty_path = os.path.join(TEST_DIR, "empty_summary.txt")
    write_summary(empty_path, "")
    report = evaluate_reflex_score(empty_path)
    assert report["step_count"] == 0
    assert report["step_scores"] == {}

def test_edge_case_non_integer_adjacency():
    summary_path = os.path.join(TEST_DIR, "edge_summary.txt")
    write_summary(summary_path, """
[🔄 Step 2 Summary]
• Influence applied: 2
• Fluid–ghost adjacents: N/A
• Pressure mutated: False
""")
    report = evaluate_reflex_score(summary_path)
    assert report["step_count"] == 1
    score = report["step_scores"].get(2)
    assert isinstance(score, float)
    assert score > 0.0
