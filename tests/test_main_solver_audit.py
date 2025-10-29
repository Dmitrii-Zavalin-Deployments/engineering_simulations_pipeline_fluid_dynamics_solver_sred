import sys
import json
import pytest
import subprocess
import importlib
from unittest import mock
from src.main_solver import run_navier_stokes_simulation


@pytest.fixture
def dummy_config():
    return {
        "domain_definition": {"nx": 1, "ny": 1, "nz": 1, "min_x": 0, "max_x": 1, "min_y": 0, "max_y": 1, "min_z": 0, "max_z": 1},
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0, 0, 0], "initial_pressure": 101325},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.01, "output_interval": 1},
        "boundary_conditions": [],
        "geometry_definition": {
            "geometry_mask_flat": [1],
            "geometry_mask_shape": [1, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }


def test_reflex_audit_import_fallback(monkeypatch, capsys, tmp_path, dummy_config):
    # ✅ Covers lines 28–31
    monkeypatch.setitem(sys.modules, "src.audit.run_reflex_audit", None)
    if "src.main_solver" in sys.modules:
        del sys.modules["src.main_solver"]
    import src.main_solver
    importlib.reload(src.main_solver)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr(src.main_solver, "load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr(src.main_solver, "validate_config", lambda config: None)
    monkeypatch.setattr(src.main_solver, "build_simulation_grid", lambda config: None)
    monkeypatch.setattr(src.main_solver, "generate_snapshots", lambda *a, **kw: [])

    src.main_solver.run_navier_stokes_simulation(str(dummy_input))
    captured = capsys.readouterr()
    assert "⚠️ Reflex audit module not available" in captured.out


def test_run_reflex_audit_executes(monkeypatch, tmp_path, dummy_config):
    # ✅ Covers lines 129–131
    called = mock.Mock()
    monkeypatch.setattr("src.main_solver.audit_available", True)
    monkeypatch.setattr("src.main_solver.run_reflex_audit", called)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [])

    run_navier_stokes_simulation(str(dummy_input))
    called.assert_called_once()


def test_run_reflex_audit_skipped(monkeypatch, tmp_path, capsys, dummy_config):
    # ✅ Covers lines 138–147
    monkeypatch.setattr("src.main_solver.audit_available", False)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [])

    run_navier_stokes_simulation(str(dummy_input))
    captured = capsys.readouterr()
    assert "⚠️ Reflex audit skipped" in captured.out


def test_run_reflex_audit_exception(monkeypatch, tmp_path, capsys, dummy_config):
    # ✅ Covers lines 129–131 with simulated failure
    def faulty_audit():
        raise RuntimeError("Simulated audit failure")

    monkeypatch.setattr("src.main_solver.audit_available", True)
    monkeypatch.setattr("src.main_solver.run_reflex_audit", faulty_audit)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr("src.main_solver.load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr("src.main_solver.validate_config", lambda config: None)
    monkeypatch.setattr("src.main_solver.build_simulation_grid", lambda config: None)
    monkeypatch.setattr("src.main_solver.generate_snapshots", lambda *a, **kw: [])

    run_navier_stokes_simulation(str(dummy_input))
    captured = capsys.readouterr()
    assert "⚠️ Reflex audit failed: Simulated audit failure" in captured.out


def test_main_solver_cli(tmp_path, dummy_config):
    # ✅ Covers lines 147–151
    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    result = subprocess.run(
        [sys.executable, "src/main_solver.py", str(dummy_input)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "✅ Simulation complete" in result.stdout
