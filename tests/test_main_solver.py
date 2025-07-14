# tests/test_main_solver.py
# 🧪 Unit tests for main_solver.py — validates CLI behavior, config handling, and snapshot writing

import os
import json
import shutil
import pytest

from src.main_solver import load_reflex_config, run_solver


@pytest.fixture
def fake_input_json(tmp_path):
    input_data = {
        "simulation_parameters": {
            "time_step": 1.0,
            "total_time": 2.0,           # ✅ ensures step 0, 1, 2
            "output_interval": 1         # ✅ ensures all steps are output
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


def test_run_solver_creates_expected_snapshot(fake_input_json):
    # Clear output folder
    output_folder = "data/testing-input-output/navier_stokes_output"
    shutil.rmtree(output_folder, ignore_errors=True)

    # Run the solver
    run_solver(fake_input_json)

    # File should match step 2 snapshot
    expected_file = os.path.join(output_folder, "fluid_simulation_input_step_0002.json")
    assert os.path.isfile(expected_file), f"❌ Missing snapshot file: {expected_file}"

    # Check key output artifacts
    assert os.path.exists(os.path.join(output_folder, "step_summary.txt"))
    assert os.path.exists(os.path.join(output_folder, "mutation_pathways_log.json"))
    assert os.path.exists(os.path.join(output_folder, "influence_flags_log.json"))



