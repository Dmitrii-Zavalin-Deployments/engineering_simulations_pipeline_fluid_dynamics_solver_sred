import pytest
import json
import os
from pathlib import Path
from src.tools import snapshot_validator

@pytest.fixture
def test_folder(tmp_path):
    folder = tmp_path / "snapshots"
    folder.mkdir()
    return folder

def write_snapshot(folder, step_index, grid, reflex):
    snap = {
        "step_index": step_index,
        "grid": grid,
        **reflex
    }
    path = folder / f"snapshot_step_{step_index:04d}.json"
    path.write_text(json.dumps(snap))
    return path

def write_influence_log(folder, entries):
    path = folder / "influence_flags_log.json"
    path.write_text(json.dumps(entries))
    return path

def write_divergence_log(folder, step_index, pre, post):
    path = folder / "divergence_log.txt"
    lines = [
        f"Step {step_index:04d} | Stage: before projection | Max: {pre:.6e} | Mean: 0.0\n",
        f"Step {step_index:04d} | Stage: after projection | Max: {post:.6e} | Mean: 0.0\n"
    ]
    folder.joinpath("divergence_log.txt").write_text("".join(lines))
    return path

def test_pressure_mutation_without_trigger_warns(test_folder, capsys):
    write_snapshot(test_folder, 1, [], {
        "pressure_mutated": True,
        "velocity_projected": True,
        "triggered_by": []
    })
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "Pressure mutated without ghost or boundary trigger" in output

def test_velocity_projected_without_mutation_warns(test_folder, capsys):
    write_snapshot(test_folder, 2, [], {
        "pressure_mutated": False,
        "velocity_projected": True,
        "triggered_by": []
    })
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "Velocity projected but no pressure mutation flagged" in output

def test_influence_flag_mismatch_warns(test_folder, capsys):
    grid = [
        {"fluid_mask": True, "influenced_by_ghost": True},
        {"fluid_mask": True, "influenced_by_ghost": False}
    ]
    write_snapshot(test_folder, 3, grid, {})
    write_influence_log(test_folder, [{"step_index": 3, "influenced_cell_count": 0}])
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "Influence flag mismatch" in output

def test_divergence_collapse_detected(test_folder, capsys):
    write_snapshot(test_folder, 4, [], {})
    write_divergence_log(test_folder, 4, pre=0.02, post=0.01)
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "Divergence decreased" in output

def test_divergence_increase_detected(test_folder, capsys):
    write_snapshot(test_folder, 5, [], {})
    write_divergence_log(test_folder, 5, pre=0.01, post=0.02)
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "Divergence increased" in output

def test_missing_divergence_log_warns(test_folder, capsys):
    write_snapshot(test_folder, 6, [], {})
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "No divergence_log.txt found" in output

def test_insufficient_divergence_entries_warns(test_folder, capsys):
    path = test_folder / "divergence_log.txt"
    path.write_text("Step 0007 | Stage: before projection | Max: 1.0e-2 | Mean: 0.0\n")
    write_snapshot(test_folder, 7, [], {})
    snapshot_validator.run_snapshot_validation(str(test_folder))
    output = capsys.readouterr().out
    assert "Not enough divergence logs to verify collapse" in output



