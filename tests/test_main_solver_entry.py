# tests/test_main_solver_entry.py

import json
import pytest
from unittest import mock
from src.main_solver import run_navier_stokes_simulation, load_reflex_config


@pytest.fixture
def full_input_config(tmp_path):
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 1
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0, 0, 0], "initial_pressure": 101325},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.01, "output_interval": 1},
        "boundary_conditions": [],
        "ghost_rules": {
            "boundary_faces": ["x_min", "x_max"],
            "default_type": "wall",
            "face_types": {"xmin": "inlet", "xmax": "outlet"}
        },
        "step_index": 0
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    return path


def test_run_simulation_with_debug(monkeypatch, full_input_config, tmp_path, capsys):
    monkeypatch.setattr("src.main_solver.debug", True)
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)

    dummy_snapshot = {"reflex_score": 4.5, "velocity_field": {}, "pressure_field": {}}
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [(0, dummy_snapshot)])
    monkeypatch.setattr("src.main_solver.compact_pressure_delta_map", lambda *a, **kw: None)
    monkeypatch.setattr("src.main_solver.run_reflex_audit", lambda: None)
    monkeypatch.setattr("src.main_solver.audit_available", True)

    run_navier_stokes_simulation(str(full_input_config), output_dir=str(tmp_path))
    captured = capsys.readouterr()
    assert "📐 Grid resolution" in captured.out
    assert "🔄 Step 0000 written" in captured.out
    assert "📉 Compacted pressure delta map" in captured.out


def test_run_simulation_skips_audit(monkeypatch, full_input_config, tmp_path, capsys):
    monkeypatch.setattr("src.main_solver.debug", True)
    monkeypatch.setattr("src.main_solver.audit_available", False)
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [])

    run_navier_stokes_simulation(str(full_input_config), output_dir=str(tmp_path))
    captured = capsys.readouterr()
    assert "⚠️ Reflex audit skipped" in captured.out


def test_load_reflex_config_fallback(tmp_path):
    config = load_reflex_config("nonexistent.yaml")
    assert config["reflex_verbosity"] == "medium"
    assert config["ghost_adjacency_depth"] == 1
