# ✅ Unit Test Suite — Main Solver
# 📄 Full Path: tests/test_main_solver.py

import pytest
import os
import json
import yaml
from tempfile import TemporaryDirectory
from src.main_solver import run_solver, load_reflex_config

def make_valid_input(filepath):
    input_data = {
        "domain_definition": {
            "min_x": 0, "max_x": 1, "nx": 1,
            "min_y": 0, "max_y": 1, "ny": 1,
            "min_z": 0, "max_z": 1, "nz": 1
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "simulation_parameters": {
            "output_interval": 1,
            "time_step": 0.01,
            "total_time": 1.0
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

def test_solver_runs_and_generates_snapshot(tmp_path, monkeypatch):
    input_path = tmp_path / "scenario.json"
    make_valid_input(input_path)

    # Patch output folder to keep outputs inside test tmp directory
    monkeypatch.setattr("src.main_solver.os.path.join",
                        lambda *args: tmp_path / "navier_stokes_output")

    run_solver(str(input_path))

    output_dir = tmp_path / "navier_stokes_output"
    assert output_dir.exists()
    files = os.listdir(output_dir)
    assert any("step_" in f and f.endswith(".json") for f in files)

def test_load_reflex_config_with_missing_file():
    fallback = load_reflex_config("nonexistent.yaml")
    assert isinstance(fallback, dict)
    assert "reflex_verbosity" in fallback

def test_load_reflex_config_with_valid_yaml(tmp_path):
    path = tmp_path / "reflex.yaml"
    config = {
        "reflex_verbosity": "high",
        "include_divergence_delta": True
    }
    with open(path, "w") as f:
        yaml.dump(config, f)
    result = load_reflex_config(str(path))
    assert result["reflex_verbosity"] == "high"
    assert result["include_divergence_delta"] is True



