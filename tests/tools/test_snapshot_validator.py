# tests/test_snapshot_validator.py
# 🧪 Unit tests for snapshot_validator.py — validates snapshot structure, reflex flags, influence and divergence consistency

import os
import tempfile
import json
import pytest
from src.tools import snapshot_validator

def create_dummy_snapshot(folder, step_index, grid=None, reflex=None):
    path = os.path.join(folder, f"snapshot_step_{step_index:04d}.json")
    snap = {
        "step_index": step_index,
        "grid": grid or [],
        "pressure_mutated": reflex.get("pressure_mutated", False),
        "triggered_by": reflex.get("triggered_by", []),
        "velocity_projected": reflex.get("velocity_projected", False)
    }
    with open(path, "w") as f:
        json.dump(snap, f)
    return path

def create_dummy_influence_log(folder, entries):
    path = os.path.join(folder, "influence_flags_log.json")
    with open(path, "w") as f:
        json.dump(entries, f)

def create_dummy_divergence_log(folder, step_index, pre, post):
    path = os.path.join(folder, "divergence_log.txt")
    with open(path, "w") as f:
        f.write(f"Step {step_index:04d} | Stage: before projection | Max: {pre:.6e} | ...\n")
        f.write(f"Step {step_index:04d} | Stage: after projection | Max: {post:.6e} | ...\n")

def test_load_snapshots_filters_correct_files():
    with tempfile.TemporaryDirectory() as folder:
        create_dummy_snapshot(folder, 0)
        create_dummy_snapshot(folder, 1)
        open(os.path.join(folder, "random_file.txt"), "w").close()
        paths = snapshot_validator.load_snapshots(folder)
        assert len(paths) == 2
        assert all("_step_" in p for p in paths)

def test_load_influence_log_returns_data():
    with tempfile.TemporaryDirectory() as folder:
        data = [{"step_index": 0, "influenced_cell_count": 2}]
        create_dummy_influence_log(folder, data)
        loaded = snapshot_validator.load_influence_log(folder)
        assert loaded == data

def test_load_influence_log_missing_returns_empty():
    with tempfile.TemporaryDirectory() as folder:
        assert snapshot_validator.load_influence_log(folder) == []

def test_validate_pressure_mutation_warns_missing_trigger(capsys):
    snap = {
        "pressure_mutated": True,
        "triggered_by": [],
        "velocity_projected": False
    }
    snapshot_validator.validate_pressure_mutation(snap, step_index=1)
    out = capsys.readouterr().out
    assert "Pressure mutated without ghost" in out

def test_validate_pressure_mutation_warns_projected_without_mutation(capsys):
    snap = {
        "pressure_mutated": False,
        "triggered_by": [],
        "velocity_projected": True
    }
    snapshot_validator.validate_pressure_mutation(snap, step_index=2)
    out = capsys.readouterr().out
    assert "Velocity projected but no pressure mutation" in out

def test_validate_influence_consistency_warns_mismatch(capsys):
    snap = {
        "grid": [
            {"fluid_mask": True, "influenced_by_ghost": True},
            {"fluid_mask": True, "influenced_by_ghost": False}
        ]
    }
    influence_log = [{"step_index": 3, "influenced_cell_count": 0}]
    snapshot_validator.validate_influence_consistency(snap, influence_log, step_index=3)
    out = capsys.readouterr().out
    assert "Influence flag mismatch" in out

def test_validate_divergence_collapse_warns_missing_log(capsys):
    with tempfile.TemporaryDirectory() as folder:
        snapshot_validator.validate_divergence_collapse(folder, step_index=4)
        out = capsys.readouterr().out
        assert "No divergence_log.txt found" in out

def test_validate_divergence_collapse_parses_and_warns(capsys):
    with tempfile.TemporaryDirectory() as folder:
        create_dummy_divergence_log(folder, step_index=5, pre=1e-2, post=1e-3)
        snapshot_validator.validate_divergence_collapse(folder, step_index=5)
        out = capsys.readouterr().out
        assert "Divergence decreased" in out

def test_validate_divergence_collapse_warns_increase(capsys):
    with tempfile.TemporaryDirectory() as folder:
        create_dummy_divergence_log(folder, step_index=6, pre=1e-3, post=1e-2)
        snapshot_validator.validate_divergence_collapse(folder, step_index=6)
        out = capsys.readouterr().out
        assert "Divergence increased" in out

def test_run_snapshot_validation_integrates_all(capsys):
    with tempfile.TemporaryDirectory() as folder:
        create_dummy_snapshot(folder, 7, grid=[{"fluid_mask": True, "influenced_by_ghost": True}],
                              reflex={"pressure_mutated": True, "velocity_projected": False})
        create_dummy_influence_log(folder, [{"step_index": 7, "influenced_cell_count": 0}])
        create_dummy_divergence_log(folder, 7, pre=0.02, post=0.01)
        snapshot_validator.run_snapshot_validation(folder)
        out = capsys.readouterr().out
        assert "Snapshot validation complete" in out
        assert "[Step 7]" in out