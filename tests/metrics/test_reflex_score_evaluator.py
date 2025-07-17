# tests/metrics/test_reflex_score_evaluator.py
# ✅ Unit tests for src/metrics/reflex_score_evaluator.py

import os
import json
import pytest
from src.metrics.reflex_score_evaluator import (
    evaluate_reflex_score,
    compute_score,
    evaluate_snapshot_health,
    batch_evaluate_trace,
    ADJACENCY_BONUS
)

@pytest.fixture
def summary_file(tmp_path):
    path = tmp_path / "step_summary.txt"
    content = """
[🔄 Step 0 Summary]
• Influence applied: 8
• Fluid–ghost adjacents: 6
• Pressure mutated: True

[🔄 Step 1 Summary]
• Influence applied: 2
• Fluid–ghost adjacents: 3
• Pressure mutated: False
• Influence suppressed: 2
"""
    path.write_text(content.strip())
    return str(path)

def test_evaluate_reflex_score_from_summary(summary_file):
    result = evaluate_reflex_score(summary_file)
    assert result["step_count"] == 2
    assert result["step_scores"][0] > result["step_scores"][1]
    assert isinstance(result["average_score"], float)
    assert "ghost_adjacency_no_mutation" in result["score_tags"][1]

def test_compute_score_with_bonus_applied():
    score, tags = compute_score({
        "influence": 1,
        "adjacency": 2,
        "mutation": False,
        "suppressed": 1
    })
    expected_base = round(0.5 * 0.1 + 0.3 * 0.2 + 0.2 * 0.0, 4)
    expected_total = round(expected_base + ADJACENCY_BONUS, 4)
    assert score == expected_total
    assert "ghost_adjacency_no_mutation" in tags

def test_compute_score_with_mutation_detected():
    score, tags = compute_score({
        "influence": 5,
        "adjacency": 2,
        "mutation": True
    })
    assert "mutation_detected" in tags
    assert 0.0 < score <= 1.0

@pytest.fixture
def delta_map_file(tmp_path):
    path = tmp_path / "pressure_delta_map_step_0000.json"
    json.dump({
        "(0.0, 0.0, 0.0)": {"delta": 0.02},
        "(1.0, 0.0, 0.0)": {"delta": 0.0}
    }, path.open("w"))
    return str(path)

@pytest.fixture
def pathway_log_file(tmp_path):
    path = tmp_path / "mutation_pathways_log.json"
    json.dump([
        {"step_index": 0},
        {"step_index": 1}
    ], path.open("w"))
    return str(path)

def test_evaluate_snapshot_health_passes(delta_map_file, pathway_log_file):
    reflex_meta = {
        "step_index": 0,
        "pressure_solver_invoked": True,
        "post_projection_divergence": 0.1,
        "reflex_score": 5
    }
    report = evaluate_snapshot_health(
        step_index=0,
        delta_map_path=delta_map_file,
        pathway_log_path=pathway_log_file,
        reflex_metadata=reflex_meta
    )
    assert report["mutated_cells"] == 1
    assert report["pathway_recorded"] is True
    assert report["has_projection"] is True
    assert report["divergence_logged"] is True
    assert report["reflex_score"] == 5

def test_batch_evaluate_trace_runs(delta_map_file, pathway_log_file):
    snapshots = [{"step_index": 0, "pressure_solver_invoked": True, "reflex_score": 3}]
    reports = batch_evaluate_trace(
        trace_folder=os.path.dirname(delta_map_file),
        pathway_log_path=pathway_log_file,
        reflex_snapshots=snapshots
    )
    assert len(reports) == 1
    assert "mutated_cells" in reports[0]



