# tests/test_main_solver.py

import os
import json
import pytest
from src.main_solver import generate_snapshots
from src.input_reader import load_simulation_input

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
        assert "step" in snap
        assert "grid" in snap
        assert "max_velocity" in snap
        assert "projection_passes" in snap

def test_snapshot_step_padding():
    formatted = [f"{i:04d}" for i in [1, 12, 123, 1234]]
    assert formatted == ["0001", "0012", "0123", "1234"]

def test_output_interval_larger_than_steps():
    input_dict = VALID_INPUT.copy()
    input_dict["simulation_parameters"]["output_interval"] = 20
    snaps = generate_snapshots(input_dict, "fluid_simulation_input")
    assert len(snaps) == 1  # Only step 0

def test_zero_output_interval_raises(input_dict):
    input_dict["simulation_parameters"]["output_interval"] = 0
    with pytest.raises(ZeroDivisionError):
        generate_snapshots(input_dict, "fail_case")



