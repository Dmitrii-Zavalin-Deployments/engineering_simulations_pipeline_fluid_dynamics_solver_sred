# tests/test_main_solver.py

import pytest
from dataclasses import asdict
from src.main_solver import generate_snapshots
from src.grid_generator import generate_grid
from src.metrics.velocity_metrics import compute_max_velocity

VALID_INPUT = {
    "domain_definition": {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 10, "ny": 10, "nz": 10
    },
    "fluid_properties": {
        "density": 1.0, "viscosity": 0.01
    },
    "initial_conditions": {
        "initial_velocity": [0.1, 0.0, 0.0],
        "initial_pressure": 100.0
    },
    "simulation_parameters": {
        "time_step": 0.1,
        "total_time": 1.0,
        "output_interval": 2
    },
    "boundary_conditions": [
        {
            "faces": [1],
            "type": "dirichlet",
            "apply_to": ["pressure"],
            "pressure": 100.0
        }
    ]
}

@pytest.fixture
def input_dict():
    return VALID_INPUT.copy()

def test_generate_snapshots_count(input_dict):
    snaps = generate_snapshots(input_dict, "fluid_simulation_input")
    assert len(snaps) == 6  # steps: 0,2,4,6,8,10

def test_snapshot_keys(input_dict):
    snaps = generate_snapshots(input_dict, "test_case")
    for step, snap in snaps:
        for key in [
            "step", "grid", "max_velocity", "max_divergence",
            "global_cfl", "overflow_detected", "damping_enabled",
            "projection_passes"
        ]:
            assert key in snap

def test_serialization_grid_dicts(input_dict):
    snaps = generate_snapshots(input_dict, "test_case")
    for _, snap in snaps:
        assert isinstance(snap["grid"], list)
        assert all(isinstance(c, dict) for c in snap["grid"])
        assert all(set(["x", "y", "z", "velocity", "pressure"]).issubset(c.keys()) for c in snap["grid"])

def test_step_formatting():
    formatted = [f"{i:04d}" for i in [0, 1, 12, 123, 1234]]
    assert formatted == ["0000", "0001", "0012", "0123", "1234"]

def test_output_interval_larger_than_total_steps():
    modified = VALID_INPUT.copy()
    modified["simulation_parameters"]["output_interval"] = 20
    snaps = generate_snapshots(modified, "fluid_simulation_input")
    assert len(snaps) == 1  # Only step 0 written

def test_zero_output_interval_triggers_exception(input_dict):
    input_dict["simulation_parameters"]["output_interval"] = 0
    with pytest.raises(ZeroDivisionError):
        generate_snapshots(input_dict, "fail_case")

def test_velocity_is_consistent(input_dict):
    grid = generate_grid(input_dict["domain_definition"], input_dict["initial_conditions"])
    velocity = input_dict["initial_conditions"]["initial_velocity"]
    max_vel = compute_max_velocity(grid)
    expected = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5
    assert round(max_vel, 5) == round(expected, 5)

def test_snapshot_contains_valid_cell_fields(input_dict):
    snaps = generate_snapshots(input_dict, "validation_check")
    for _, snap in snaps:
        for cell in snap["grid"]:
            assert isinstance(cell["x"], (int, float))
            assert isinstance(cell["y"], (int, float))
            assert isinstance(cell["z"], (int, float))
            assert isinstance(cell["velocity"], list)
            assert isinstance(cell["pressure"], (int, float))
            assert len(cell["velocity"]) == 3



