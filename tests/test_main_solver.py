# tests/test_main_solver.py
# 🧪 Unit tests for main_solver.py — validates CLI behavior, config handling, and snapshot writing

import os
import json
import tempfile
import shutil
import pytest

from src.main_solver import load_reflex_config, run_solver


@pytest.fixture
def fake_input_json(tmp_path):
    input_data = {
        "simulation_parameters": {
            "time_step": 1.0,
            "total_time": 1.0,
            "output_interval": 1
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0
        }
    }
    file_path = tmp_path / "fluid_simulation_input.json"
    with open(file_path, "w") as f:
        json.dump(input_data, f)
    return str(file_path)


def test_load_reflex_config_defaults_gracefully():
    config = load_reflex_config("nonexistent/config.yaml")
    assert isinstance(config, dict)
    assert "ghost_adjacency_depth" in config


def test_run_solver_creates_outputs(fake_input_json, monkeypatch):
    # Clear output folder
    output_folder = "data/testing-input-output/navier_stokes_output"
    shutil.rmtree(output_folder, ignore_errors=True)

    # Patch sys.argv to simulate CLI call
    monkeypatch.setattr("sys.argv", ["main_solver.py", fake_input_json])

    # Run
    from src.main_solver import run_solver
    run_solver(fake_input_json)

    # Check snapshot files created
    snapshot_files = os.listdir(output_folder)
    assert any(f.endswith(".json") and "fluid_simulation_input" in f for f in snapshot_files)

    # Check summary and logs
    assert os.path.exists(os.path.join(output_folder, "step_summary.txt"))
    assert os.path.exists(os.path.join(output_folder, "mutation_pathways_log.json"))
    assert os.path.exists(os.path.join(output_folder, "influence_flags_log.json"))



