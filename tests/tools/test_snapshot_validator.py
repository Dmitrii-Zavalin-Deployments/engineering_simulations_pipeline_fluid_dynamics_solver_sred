# tests/tools/test_snapshot_validator.py
# 🧪 Unit tests for snapshot_validator.py — validates consistency checks and edge-case robustness

import os
import json
import tempfile
from src.tools import snapshot_validator

def test_validate_pressure_mutation_flags(capsys):
    reflex = {
        "pressure_mutated": True,
        "velocity_projected": True,
        "triggered_by": []  # Missing ghost/boundary
    }
    snapshot_validator.validate_pressure_mutation(reflex, step_index=0)
    output = capsys.readouterr().out
    assert "Pressure mutated without ghost or boundary trigger" in output

def test_validate_pressure_mutation_flags_skipped(capsys):
    reflex = {
        "pressure_mutated": False,
        "velocity_projected": True,
        "triggered_by": ["ghost_influence"]
    }
    snapshot_validator.validate_pressure_mutation(reflex, step_index=1)
    output = capsys.readouterr().out
    assert "Velocity projected but no pressure mutation flagged" in output

def test_validate_influence_consistency_match(capsys):
    snapshot = {
        "grid": [
            {"fluid_mask": True, "influenced_by_ghost": True},
            {"fluid_mask": True, "influenced_by_ghost": False}
        ]
    }
    influence_log = [{"step_index": 2, "influenced_cell_count": 1}]
    snapshot_validator.validate_influence_consistency(snapshot, influence_log, step_index=2)
    output = capsys.readouterr().out
    assert output == ""  # Silent pass

def test_validate_influence_consistency_mismatch(capsys):
    snapshot = {
        "grid": [{"fluid_mask": True, "influenced_by_ghost": True}]
    }
    influence_log = [{"step_index": 3, "influenced_cell_count": 0}]
    snapshot_validator.validate_influence_consistency(snapshot, influence_log, step_index=3)
    output = capsys.readouterr().out
    assert "Influence flag mismatch" in output

def test_validate_divergence_collapse_increase(tmp_path, capsys):
    log_path = tmp_path / "divergence_log.txt"
    log_path.write_text(
        "Step 0004 | Stage: before projection | Max: 1.0e-3 | Mean: ... | Count: ...\n"
        "Step 0004 | Stage: after projection  | Max: 2.0e-3 | Mean: ... | Count: ...\n"
    )
    snapshot_validator.validate_divergence_collapse(tmp_path, step_index=4)
    output = capsys.readouterr().out
    assert "Divergence increased" in output

def test_validate_divergence_collapse_decrease(tmp_path, capsys):
    log_path = tmp_path / "divergence_log.txt"
    log_path.write_text(
        "Step 0005 | Stage: before projection | Max: 3.0e-3 | Mean: ... | Count: ...\n"
        "Step 0005 | Stage: after projection  | Max: 2.0e-3 | Mean: ... | Count: ...\n"
    )
    snapshot_validator.validate_divergence_collapse(tmp_path, step_index=5)
    output = capsys.readouterr().out
    assert "Divergence decreased" in output

def test_validate_divergence_collapse_missing_log(tmp_path, capsys):
    snapshot_validator.validate_divergence_collapse(tmp_path, step_index=99)
    output = capsys.readouterr().out
    assert "No divergence_log.txt found" in output

def test_run_snapshot_validation_end_to_end(tmp_path, capsys):
    # Create dummy snapshot and logs
    folder = tmp_path
    snap_file = folder / "scenario_step_0000.json"
    snap_file.write_text(json.dumps({
        "step_index": 0,
        "pressure_mutated": True,
        "velocity_projected": True,
        "triggered_by": [],
        "grid": [{"fluid_mask": True, "influenced_by_ghost": True}]
    }))

    influence_log = folder / "influence_flags_log.json"
    influence_log.write_text(json.dumps([{
        "step_index": 0,
        "influenced_cell_count": 0
    }]))

    div_log = folder / "divergence_log.txt"
    div_log.write_text(
        "Step 0000 | Stage: before projection | Max: 1.5e-3 | Mean: ... | Count: ...\n"
        "Step 0000 | Stage: after projection  | Max: 2.0e-3 | Mean: ... | Count: ...\n"
    )

    snapshot_validator.run_snapshot_validation(str(folder))
    output = capsys.readouterr().out
    assert "Pressure mutated without ghost or boundary trigger" in output
    assert "Influence flag mismatch" in output
    assert "Divergence increased" in output



