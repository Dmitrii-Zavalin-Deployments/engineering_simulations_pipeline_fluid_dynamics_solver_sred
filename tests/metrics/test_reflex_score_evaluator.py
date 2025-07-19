# tests/metrics/test_reflex_score_evaluator.py
# 🧪 Unit tests for src/metrics/reflex_score_evaluator.py

import os
import tempfile
import json
from src.metrics.reflex_score_evaluator import (
    evaluate_reflex_score,
    compute_score,
    evaluate_snapshot_health,
    batch_evaluate_trace,
)

def test_compute_score_with_mutation_and_influence():
    inputs = {"mutation": True, "influence": 5, "adjacency": 3}
    assert compute_score(inputs) == 2.0

def test_compute_score_with_mutation_and_adjacency_only():
    inputs = {"mutation": True, "influence": 0, "adjacency": 2}
    assert compute_score(inputs) == 0.2

def test_compute_score_with_mutation_and_no_ghost():
    inputs = {"mutation": True, "influence": 0, "adjacency": 0}
    assert compute_score(inputs) == 0.2

def test_compute_score_with_no_mutation():
    inputs = {"mutation": False, "influence": 5, "adjacency": 2}
    assert compute_score(inputs) == 0.0

def test_evaluate_reflex_score_from_summary_lines():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("[🔄 Step 0001 Summary]\n")
        f.write("Influence applied: 5\n")
        f.write("Fluid–ghost adjacents: 2\n")
        f.write("Pressure mutated: True\n")
        f.write("[🔄 Step 0002 Summary]\n")
        f.write("Influence applied: 0\n")
        f.write("Fluid–ghost adjacents: 0\n")
        f.write("Pressure mutated: True\n")
        f.flush()
        f_path = f.name

    result = evaluate_reflex_score(f_path)
    os.remove(f_path)

    assert result["step_scores"][1] == 2.0
    assert result["step_scores"][2] == 0.2
    assert result["step_count"] == 2
    assert result["max_score"] == 2.0
    assert result["min_score"] == 0.2

def test_evaluate_snapshot_health_returns_expected_structure(tmp_path):
    delta_path = tmp_path / "delta.json"
    trace_path = tmp_path / "trace.json"

    json.dump({"c1": {"delta": 0.0}, "c2": {"delta": 1.2}}, open(delta_path, "w"))
    json.dump([{"step_index": 3}], open(trace_path, "w"))

    reflex = {
        "step_index": 3,
        "pressure_solver_invoked": True,
        "post_projection_divergence": 0.9,
        "reflex_score": 1.5
    }

    result = evaluate_snapshot_health(
        step_index=3,
        delta_map_path=str(delta_path),
        pathway_log_path=str(trace_path),
        reflex_metadata=reflex
    )

    assert result["step_index"] == 3
    assert result["mutated_cells"] == 1
    assert result["pathway_recorded"] is True
    assert result["has_projection"] is True
    assert result["divergence_logged"] is True
    assert result["reflex_score"] == 1.5

def test_batch_evaluate_trace_returns_list(tmp_path):
    trace_dir = tmp_path
    delta_file = trace_dir / "pressure_delta_map_step_0001.json"
    trace_file = trace_dir / "trace_log.json"

    json.dump({"cellA": {"delta": 0.3}}, open(delta_file, "w"))
    json.dump([{"step_index": 1}], open(trace_file, "w"))

    reflex_snapshot = [{"step_index": 1, "pressure_solver_invoked": False, "reflex_score": 0.0}]
    reports = batch_evaluate_trace(str(trace_dir), str(trace_file), reflex_snapshot)

    assert isinstance(reports, list)
    assert reports[0]["step_index"] == 1
    assert reports[0]["mutated_cells"] == 1



