# ✅ Unit Test Suite — Main Solver (Fully Patched)
# 📄 Full Path: tests/test_main_solver.py

import pytest
import os
import json
import yaml
from tempfile import TemporaryDirectory
from src.main_solver import run_solver, load_reflex_config

def make_valid_input(filepath):
    input_data = {
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0], "initial_pressure": 0.0},
        "simulation_parameters": {
            "output_interval": 1,
            "time_step": 0.01,
            "total_time": 1.0  # ✅ Patched to prevent KeyError
        },
        "boundary_conditions": {
            "apply_to": ["x-min"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
            "no_slip": True
        },
        "pressure_solver": {"method": "jacobi", "tolerance": 1e-6}
    }
    with open(filepath, "w") as f:
        json.dump(input_data, f)

def test_solver_runs_and_generates_snapshot(tmp_path):
    input_path = tmp_path / "scenario.json"
    make_valid_input(input_path)
    run_solver(str(input_path))

    output_dir = tmp_path / "testing-input-output" / "navier_stokes_output"
    assert os.path.exists(output_dir)
    files = os.listdir(output_dir)
    snapshot_files = [f for f in files if f.startswith("scenario_step_")]
    assert len(snapshot_files) >= 1

def test_load_reflex_config_with_missing_file():
    fallback = load_reflex_config("nonexistent.yaml")
    assert isinstance(fallback, dict)
    assert "reflex_verbosity" in fallback

def test_load_reflex_config_with_valid_yaml(tmp_path):
    path = tmp_path / "reflex.yaml"
    config = {"reflex_verbosity": "high", "include_divergence_delta": True}
    with open(path, "w") as f:
        yaml.dump(config, f)
    result = load_reflex_config(str(path))
    assert result["reflex_verbosity"] == "high"
    assert result["include_divergence_delta"] is True



