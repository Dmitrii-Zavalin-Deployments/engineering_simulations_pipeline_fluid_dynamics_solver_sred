import os
import json
import pytest
import builtins
import pathlib
import tempfile
from unittest import mock
from src.main_solver import run_navier_stokes_simulation, load_reflex_config

@pytest.fixture
def mock_input_file(tmp_path):
    input_data = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 1
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0], "initial_pressure": 101325.0},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.1, "output_interval": 0.05},
        "boundary_conditions": [],
        "geometry_definition": {
            "geometry_mask_flat": [1, 1, 1, 1],
            "geometry_mask_shape": [2, 2, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(input_data))
    return path

def test_load_reflex_config_valid(tmp_path):
    config_path = tmp_path / "reflex.yaml"
    config_path.write_text("reflex_verbosity: high\ninclude_divergence_delta: true")
    config = load_reflex_config(str(config_path))
    assert config["reflex_verbosity"] == "high"
    assert config["include_divergence_delta"] is True

def test_load_reflex_config_fallback():
    config = load_reflex_config("nonexistent.yaml")
    assert config["reflex_verbosity"] == "medium"
    assert config["ghost_adjacency_depth"] == 1

def test_run_simulation_triggers_all(monkeypatch, mock_input_file, tmp_path):
    # Patch all downstream calls
    monkeypatch.setattr("src.main_solver.load_simulation_input", lambda path: json.loads(mock_input_file.read_text()))
    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda domain_path, ghost_path, step_index: json.loads(mock_input_file.read_text()))
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)

    dummy_snapshot = {
        "reflex_score": 4.2,
        "velocity_field": {},
        "pressure_field": {}
    }
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda config, scenario_name, config=None: [(0, dummy_snapshot)])
    monkeypatch.setattr("src.main_solver.compact_pressure_delta_map", lambda orig, comp: None)
    monkeypatch.setattr("src.main_solver.run_reflex_audit", lambda: None)

    output_dir = tmp_path / "output"
    run_navier_stokes_simulation(str(mock_input_file), output_dir=str(output_dir))

    saved = list(output_dir.glob("*.json"))
    assert len(saved) == 1
    assert "step_0000" in saved[0].name
    content = json.loads(saved[0].read_text())
    assert content["reflex_score"] == 4.2

def test_run_simulation_skips_compaction(monkeypatch, mock_input_file, tmp_path):
    monkeypatch.setattr("src.main_solver.load_simulation_input", lambda path: json.loads(mock_input_file.read_text()))
    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda domain_path, ghost_path, step_index: json.loads(mock_input_file.read_text()))
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)

    dummy_snapshot = {
        "reflex_score": 3.5,
        "velocity_field": {},
        "pressure_field": {}
    }
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda config, scenario_name, config=None: [(0, dummy_snapshot)])

    compact_called = mock.Mock()
    monkeypatch.setattr("src.main_solver.compact_pressure_delta_map", compact_called)
    monkeypatch.setattr("src.main_solver.run_reflex_audit", lambda: None)

    output_dir = tmp_path / "output"
    run_navier_stokes_simulation(str(mock_input_file), output_dir=str(output_dir))

    compact_called.assert_not_called()



