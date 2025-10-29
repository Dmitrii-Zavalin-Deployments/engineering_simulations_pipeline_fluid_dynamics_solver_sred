import sys
import json
import pytest
import subprocess
import importlib
from unittest import mock

@pytest.fixture
def dummy_config():
    return {
        "domain_definition": {
            "nx": 1, "ny": 1, "nz": 1,
            "min_x": 0, "max_x": 1,
            "min_y": 0, "max_y": 1,
            "min_z": 0, "max_z": 1
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0, 0, 0], "initial_pressure": 101325},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.01, "output_interval": 1},
        "boundary_conditions": [
            {
                "role": "inlet",
                "type": "dirichlet",
                "faces": [0],
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 101325,
                "apply_faces": ["x_min"]
            }
        ],
        "geometry_definition": {
            "geometry_mask_flat": [1],
            "geometry_mask_shape": [1, 1, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }

def test_reflex_audit_import_fallback(monkeypatch, capsys, tmp_path, dummy_config):
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
    import src.main_solver
    monkeypatch.setattr(src.main_solver, "audit_available", True)
    monkeypatch.setattr(src.main_solver, "run_reflex_audit", mock.Mock())

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr(src.main_solver, "load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr(src.main_solver, "validate_config", lambda config: None)
    monkeypatch.setattr(src.main_solver, "build_simulation_grid", lambda config: None)
    monkeypatch.setattr(src.main_solver, "generate_snapshots", lambda *a, **kw: [])

    src.main_solver.run_navier_stokes_simulation(str(dummy_input))
    src.main_solver.run_reflex_audit.assert_called_once()

def test_run_reflex_audit_exception(monkeypatch, tmp_path, capsys, dummy_config):
    import src.main_solver
    def faulty_audit():
        raise RuntimeError("Simulated audit failure")

    monkeypatch.setattr(src.main_solver, "audit_available", True)
    monkeypatch.setattr(src.main_solver, "run_reflex_audit", faulty_audit)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr(src.main_solver, "load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr(src.main_solver, "validate_config", lambda config: None)
    monkeypatch.setattr(src.main_solver, "build_simulation_grid", lambda config: None)
    monkeypatch.setattr(src.main_solver, "generate_snapshots", lambda *a, **kw: [])

    src.main_solver.run_navier_stokes_simulation(str(dummy_input))
    captured = capsys.readouterr()
    assert "⚠️ Reflex audit failed: Simulated audit failure" in captured.out

def test_run_reflex_audit_skipped(monkeypatch, tmp_path, capsys, dummy_config):
    import src.main_solver
    monkeypatch.setattr(src.main_solver, "audit_available", False)

    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    monkeypatch.setattr(src.main_solver, "load_simulation_config", lambda *a, **kw: dummy_config)
    monkeypatch.setattr(src.main_solver, "validate_config", lambda config: None)
    monkeypatch.setattr(src.main_solver, "build_simulation_grid", lambda config: None)
    monkeypatch.setattr(src.main_solver, "generate_snapshots", lambda *a, **kw: [])

    src.main_solver.run_navier_stokes_simulation(str(dummy_input))
    captured = capsys.readouterr()
    assert "⚠️ Reflex audit skipped" in captured.out

def test_main_solver_cli(tmp_path, dummy_config):
    dummy_input = tmp_path / "input.json"
    dummy_input.write_text(json.dumps(dummy_config))

    result = subprocess.run(
        [sys.executable, "src/main_solver.py", str(dummy_input)],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "✅ Simulation complete" in result.stdout
