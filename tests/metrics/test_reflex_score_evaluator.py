# tests/metrics/test_reflex_score_evaluator.py
# ✅ Validation suite for src/metrics/reflex_score_evaluator.py

import pytest
import tempfile
import os
import json
from src.metrics.reflex_score_evaluator import (
    compute_score,
    evaluate_reflex_score,
    evaluate_snapshot_health,
    score_pressure_mutation_volume,
    score_mutation_pathway_presence,
    score_reflex_metadata_fields,
    batch_evaluate_trace
)

def test_compute_score_mutation_with_influence():
    inputs = {
        "mutation": True,
        "influence": 3,
        "adjacency": 0,
        "suppression": 0,
        "mutation_density": 0.3,
        "projection_passes": 4,
        "triggered_by": ["ghost_influence"],
        "boundary_mutation_ratio": 0.1
    }
    score = compute_score(inputs)
    assert score > 2.0

def test_compute_score_mutation_with_adjacency_only():
    inputs = {
        "mutation": True,
        "influence": 0,
        "adjacency": 2,
        "suppression": 0,
        "mutation_density": 0.0,
        "projection_passes": 1,
        "triggered_by": [],
        "boundary_mutation_ratio": 0.0
    }
    score = compute_score(inputs)
    assert score == 0.2

def test_compute_score_penalizes_boundary_mutation_with_suppression():
    inputs = {
        "mutation": True,
        "influence": 0,
        "adjacency": 0,
        "suppression": 2,
        "mutation_density": 0.0,
        "projection_passes": 1,
        "triggered_by": [],
        "boundary_mutation_ratio": 0.6
    }
    score = compute_score(inputs)
    assert score < 0.1

def test_compute_score_penalizes_suppression_without_mutation():
    inputs = {
        "mutation": False,
        "influence": 0,
        "adjacency": 0,
        "suppression": 3,
        "mutation_density": 0.0,
        "projection_passes": 1,
        "triggered_by": [],
        "boundary_mutation_ratio": 0.0
    }
    score = compute_score(inputs)
    assert score == 0.0

def test_score_pressure_mutation_volume_counts_deltas():
    delta_map = {
        "cell1": {"delta": 0.0},
        "cell2": {"delta": 1.2},
        "cell3": {"delta": -0.5}
    }
    assert score_pressure_mutation_volume(delta_map) == 2

def test_score_mutation_pathway_presence_detects_step():
    trace = [{"step_index": 5}, {"step_index": 10}]
    assert score_mutation_pathway_presence(trace, 10) is True
    assert score_mutation_pathway_presence(trace, 3) is False

def test_score_reflex_metadata_fields_extracts_all_keys():
    reflex = {
        "pressure_solver_invoked": True,
        "post_projection_divergence": 0.01,
        "reflex_score": 2.5,
        "suppression_zones": [1, 2],
        "mutation_density": 0.2,
        "projection_passes": 3,
        "adjacency_zones": [1],
        "triggered_by": ["ghost_influence"],
        "boundary_mutation_ratio": 0.4
    }
    result = score_reflex_metadata_fields(reflex)
    assert result["has_projection"] is True
    assert result["divergence_logged"] is True
    assert result["reflex_score"] == 2.5
    assert result["suppression_zone_count"] == 2
    assert result["mutation_density"] == 0.2
    assert result["projection_passes"] == 3
    assert result["adjacency_count"] == 1
    assert result["triggered_by"] == ["ghost_influence"]
    assert result["boundary_mutation_ratio"] == 0.4

def test_evaluate_reflex_score_parses_summary_file():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("[🔄 Step 1 Summary]\nInfluence applied: 2\nFluid–ghost adjacents: 1\nPressure mutated: True\nSuppression zones: 0\n")
        f.write("[🔄 Step 2 Summary]\nInfluence applied: 0\nFluid–ghost adjacents: 0\nPressure mutated: False\nSuppression zones: 2\n")
        f.flush()
        path = f.name

    result = evaluate_reflex_score(path)
    os.remove(path)

    assert result["step_count"] == 2
    assert result["max_score"] > result["min_score"]
    assert result["average_score"] > 0.0

def test_evaluate_snapshot_health_returns_expected_structure():
    with tempfile.NamedTemporaryFile("w+", delete=False) as delta_file:
        json.dump({"c1": {"delta": 1.0}, "c2": {"delta": 0.0}}, delta_file)
        delta_file.flush()

    with tempfile.NamedTemporaryFile("w+", delete=False) as trace_file:
        json.dump([{"step_index": 42}], trace_file)
        trace_file.flush()

    reflex_metadata = {
        "pressure_mutated": True,
        "ghost_influence_count": 1,
        "suppression_zones": [1],
        "mutation_density": 0.3,
        "projection_passes": 2,
        "adjacency_zones": [1, 2],
        "triggered_by": ["ghost_influence"],
        "boundary_mutation_ratio": 0.1,
        "pressure_solver_invoked": True,
        "post_projection_divergence": 0.01
    }

    result = evaluate_snapshot_health(
        step_index=42,
        delta_map_path=delta_file.name,
        pathway_log_path=trace_file.name,
        reflex_metadata=reflex_metadata
    )

    os.remove(delta_file.name)
    os.remove(trace_file.name)

    assert result["step_index"] == 42
    assert result["mutated_cells"] == 1
    assert result["pathway_recorded"] is True
    assert result["has_projection"] is True
    assert result["divergence_logged"] is True
    assert result["reflex_score"] > 0.0
    assert result["suppression_zone_count"] == 1
    assert result["adjacency_count"] == 2



