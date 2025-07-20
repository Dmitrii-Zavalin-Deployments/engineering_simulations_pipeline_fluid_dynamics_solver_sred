# ✅ Unit Test Suite — Snapshot Validator
# 📄 Full Path: tests/tools/test_snapshot_validator.py

import pytest
import os
import json
from tempfile import TemporaryDirectory
from src.tools import snapshot_validator as sv

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def write_lines(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)

def test_load_snapshots_filters_and_sorts():
    with TemporaryDirectory() as tmp:
        valid_names = ["ref_step_0001.json", "data_step_0002.json"]
        invalid_names = ["data.json", "meta.txt", "step_003.log"]
        for name in valid_names + invalid_names:
            open(os.path.join(tmp, name), "w").write("[]")
        result = sv.load_snapshots(tmp)
        assert isinstance(result, list)
        assert all("_step_" in r and r.endswith(".json") for r in result)
        assert sorted(result) == result

def test_load_influence_log_fallback_if_missing():
    with TemporaryDirectory() as tmp:
        result = sv.load_influence_log(tmp)
        assert result == []

def test_load_influence_log_parses_valid_file():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "influence_flags_log.json")
        write_json(path, [{"step_index": 1, "influenced_cell_count": 3}])
        result = sv.load_influence_log(tmp)
        assert isinstance(result, list)
        assert result[0]["step_index"] == 1

def test_validate_pressure_mutation_warns_missing_trigger(capsys):
    reflex = {
        "pressure_mutated": True,
        "triggered_by": [],
        "velocity_projected": False
    }
    sv.validate_pressure_mutation(reflex, step_index=12)
    captured = capsys.readouterr().out
    assert "Pressure mutated without ghost or boundary trigger" in captured

def test_validate_pressure_mutation_warns_unflagged_mutation(capsys):
    reflex = {
        "pressure_mutated": False,
        "triggered_by": ["ghost_influence"],
        "velocity_projected": True
    }
    sv.validate_pressure_mutation(reflex, step_index=7)
    captured = capsys.readouterr().out
    assert "Velocity projected but no pressure mutation flagged" in captured

def test_validate_influence_consistency_warns_on_mismatch(capsys):
    snapshot = {
        "grid": [
            {"fluid_mask": True, "influenced_by_ghost": True},
            {"fluid_mask": True, "influenced_by_ghost": True}
        ]
    }
    log = [{"step_index": 99, "influenced_cell_count": 1}]
    sv.validate_influence_consistency(snapshot, log, step_index=99)
    captured = capsys.readouterr().out
    assert "Influence flag mismatch" in captured

def test_validate_divergence_collapse_handles_missing_log(capsys):
    with TemporaryDirectory() as tmp:
        sv.validate_divergence_collapse(tmp, step_index=0)
        captured = capsys.readouterr().out
        assert "No divergence_log.txt found." in captured

def test_validate_divergence_collapse_warns_on_insufficient_steps(capsys):
    with TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "divergence_log.txt")
        write_lines(log_path, ["Step 0000 | Stage: before projection | Max: 1.00e-01 | Min: -1.00e-01\n"])
        sv.validate_divergence_collapse(tmp, step_index=0)
        captured = capsys.readouterr().out
        assert "Not enough divergence logs" in captured

def test_validate_divergence_collapse_detects_reduction(capsys):
    with TemporaryDirectory() as tmp:
        log_path = os.path.join(tmp, "divergence_log.txt")
        write_lines(log_path, [
            "Step 0001 | Stage: before projection | Max: 2.00e-01 | Min: -2.00e-01\n",
            "Step 0001 | Stage: after projection | Max: 1.00e-01 | Min: -1.00e-01\n"
        ])
        sv.validate_divergence_collapse(tmp, step_index=1)
        captured = capsys.readouterr().out
        assert "Divergence decreased" in captured

def test_run_snapshot_validation_executes_all(monkeypatch, capsys):
    with TemporaryDirectory() as tmp:
        snap = {
            "step_index": 2,
            "grid": [{"fluid_mask": True, "influenced_by_ghost": True}],
            "pressure_mutated": True,
            "velocity_projected": False,
            "triggered_by": []
        }
        write_json(os.path.join(tmp, "ref_step_0002.json"), snap)
        write_json(os.path.join(tmp, "influence_flags_log.json"), [{"step_index": 2, "influenced_cell_count": 0}])
        write_lines(os.path.join(tmp, "divergence_log.txt"), [
            "Step 0002 | Stage: before projection | Max: 1.00e-01 | Min: -1.00e-01\n",
            "Step 0002 | Stage: after projection | Max: 1.00e-01 | Min: -1.00e-01\n"
        ])
        sv.run_snapshot_validation(tmp)
        captured = capsys.readouterr().out
        assert "Validating snapshots in" in captured
        assert "Pressure mutated without ghost" in captured
        assert "Influence flag mismatch" in captured
        assert "Divergence unchanged" in captured



