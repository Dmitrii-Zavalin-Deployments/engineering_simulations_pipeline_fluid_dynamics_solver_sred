# ✅ Unit Test Suite — Snapshot Manager
# 📄 Full Path: tests/test_snapshot_manager.py

import pytest
import os
from src.snapshot_manager import generate_snapshots

@pytest.fixture
def minimal_input_data():
    return {
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 0.2,
            "output_interval": 1
        },
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0, "nx": 1,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 0.0
        },
        "boundary_conditions": {
            "apply_to": ["x-min"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
            "no_slip": True
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01
        },
        "pressure_solver": {
            "method": "jacobi",
            "tolerance": 1e-6
        }
    }

def test_generate_snapshots_returns_valid_list(minimal_input_data):
    config = {
        "reflex_verbosity": "low",
        "include_divergence_delta": True
    }
    snapshots = generate_snapshots(minimal_input_data, "test_scenario", config)
    assert isinstance(snapshots, list)
    assert len(snapshots) > 0
    for step, snap in snapshots:
        assert isinstance(step, int)
        assert isinstance(snap, dict)
        assert "step_index" in snap
        assert "grid" in snap
        assert "reflex_score" in snap

def test_output_interval_zero_warns_and_defaults(minimal_input_data, caplog):
    minimal_input_data["simulation_parameters"]["output_interval"] = 0
    config = {}
    with caplog.at_level("WARNING"):
        snapshots = generate_snapshots(minimal_input_data, "interval_fallback", config)
    assert "Using fallback of 1" in caplog.text
    assert len(snapshots) > 0

def test_generate_snapshots_skips_steps_not_in_interval(minimal_input_data):
    minimal_input_data["simulation_parameters"]["total_time"] = 0.3
    minimal_input_data["simulation_parameters"]["time_step"] = 0.1
    minimal_input_data["simulation_parameters"]["output_interval"] = 2
    config = {}
    snapshots = generate_snapshots(minimal_input_data, "interval_test", config)
    expected_steps = [0, 2]
    actual_steps = [s[0] for s in snapshots]
    assert actual_steps == expected_steps



