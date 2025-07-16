# tests/test_main_solver.py
# 🧪 Validates top-level solver orchestration, reflex config loading, and snapshot writing

import os
import sys
import json
import tempfile
import yaml
import builtins
import pytest
import types
from src.main_solver import run_solver, load_reflex_config

@pytest.fixture
def dummy_input_data():
    return {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1},
        "fluid_properties": {"viscosity": 0.5},
        "initial_conditions": {"velocity": [1.0, 0.0, 0.0], "pressure": 5.0},
        "simulation_parameters": {"time_step": 0.1, "output_interval": 10},
        "boundary_conditions": {
            "apply_to": ["velocity", "pressure"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 99.0,
            "no_slip": True
        },
        "pressure_solver": {"method": "jacobi", "tolerance": 1e-6}
    }

def mock_generate_snapshots(input_data, scenario_name, config=None):
    snapshot = {
        "step_index": 0,
        "grid": [],
        "max_velocity": 0.0,
        "global_cfl": 0.0
    }
    return [(0, snapshot), (1, snapshot)]

def test_load_reflex_config_returns_defaults_on_missing():
    result = load_reflex_config("nonexistent.yaml")
    assert isinstance(result, dict)
    assert "reflex_verbosity" in result
    assert result["reflex_verbosity"] == "medium"

def test_load_reflex_config_parses_yaml_success(tmp_path):
    cfg = {"reflex_verbosity": "high", "include_divergence_delta": True}
    path = tmp_path / "reflex_debug_config.yaml"
    path.write_text(yaml.dump(cfg))
    result = load_reflex_config(str(path))
    assert result["reflex_verbosity"] == "high"
    assert result["include_divergence_delta"] is True

def test_run_solver_writes_snapshots(tmp_path, monkeypatch, dummy_input_data, capsys):
    # Write dummy input file
    input_path = tmp_path / "fluid_simulation_input.json"
    input_path.write_text(json.dumps(dummy_input_data))

    # Patch snapshot generation
    monkeypatch.setitem(sys.modules, "src.snapshot_manager", types.SimpleNamespace(generate_snapshots=mock_generate_snapshots))
    monkeypatch.setattr("src.main_solver.generate_snapshots", mock_generate_snapshots)

    # Run solver
    run_solver(str(input_path))

    output_folder = os.path.join("data", "testing-input-output", "navier_stokes_output")
    expected_files = [
        f"{input_path.stem}_step_0000.json",
        f"{input_path.stem}_step_0001.json"
    ]
    for name in expected_files:
        path = os.path.join(output_folder, name)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
            assert isinstance(data, dict)
            assert "step_index" in data

    out = capsys.readouterr().out
    assert "Starting simulation for" in out
    assert "Step 0000 written" in out
    assert "Simulation complete" in out

def test_main_solver_exit_on_missing_arg(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main_solver.py"])
    with pytest.raises(SystemExit):
        __import__("src.main_solver")
    out = capsys.readouterr().out
    assert "Please provide an input file path" in out



