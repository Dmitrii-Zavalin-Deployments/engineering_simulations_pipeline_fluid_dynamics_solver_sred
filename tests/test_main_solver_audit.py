# tests/test_main_solver_audit.py

import sys
import json
import pytest
from unittest import mock
from src.main_solver import run_navier_stokes_simulation


def test_reflex_audit_import_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.audit.run_reflex_audit", None)
    monkeypatch.setattr("src.main_solver.audit_available", False)
    from src.main_solver import audit_available
    assert audit_available is False


def test_run_reflex_audit_executes(monkeypatch, tmp_path):
    called = mock.Mock()
    monkeypatch.setattr("src.main_solver.audit_available", True)
    monkeypatch.setattr("src.main_solver.run_reflex_audit", called)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps({
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0, 0, 0], "initial_pressure": 101325},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.01, "output_interval": 0.01},
        "boundary_conditions": [],
        "geometry_definition": {
            "geometry_mask_flat": [1],
            "geometry_mask_shape": [1, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }))

    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda *a, **kw: {})
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [])
    run_navier_stokes_simulation(str(dummy_input))
    called.assert_called_once()


def test_run_reflex_audit_skipped(monkeypatch, tmp_path):
    monkeypatch.setattr("src.main_solver.audit_available", False)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps({
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0, 0, 0], "initial_pressure": 101325},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.01, "output_interval": 0.01},
        "boundary_conditions": [],
        "geometry_definition": {
            "geometry_mask_flat": [1],
            "geometry_mask_shape": [1, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }))

    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda *a, **kw: {})
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [])
    run_navier_stokes_simulation(str(dummy_input))
