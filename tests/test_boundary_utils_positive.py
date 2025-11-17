import pytest
from src.step_2_time_stepping_loop.boundary_utils import enforce_boundary, BoundaryConditionError

# Shared config for positive tests
CONFIG = {
    "boundary_conditions": [
        {
            "role": "inlet",
            "apply_to": ["velocity", "pressure"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 133,
        },
        {
            "role": "outlet",
            "apply_to": ["pressure"],
            "pressure": 120,
        },
        {
            "role": "wall",
            "apply_to": ["velocity"],
            "velocity": [0.0, 0.0, 0.0],
        },
    ]
}

BASE_STATE = {
    "pressure": 100.0,
    "velocity": {"vx": 0.5, "vy": 0.1, "vz": -0.2},
}

# --- Positive validation tests (happy paths) ---

def test_inlet_applies_velocity_and_pressure_correctly():
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result["pressure"] == 133
    assert result["velocity"] == {"vx": 1.0, "vy": 0.0, "vz": 0.0}

def test_outlet_applies_pressure_only_correctly():
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result["pressure"] == 120
    # velocity unchanged
    assert result["velocity"] == BASE_STATE["velocity"]

def test_wall_applies_velocity_only_correctly():
    cell = {"boundary_role": "wall"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result["velocity"] == {"vx": 0.0, "vy": 0.0, "vz": 0.0}
    # pressure unchanged
    assert result["pressure"] == BASE_STATE["pressure"]

def test_no_boundary_role_returns_state_unchanged():
    cell = {"boundary_role": None}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result == BASE_STATE

# --- Additional coverage for missed lines (42, 63) ---

def test_role_none_branch_is_covered():
    # Line 42: role is None → return unchanged
    cell = {"boundary_role": None}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result == BASE_STATE

def test_pressure_override_branch_is_covered():
    # Line 63: pressure override applied
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result["pressure"] == 120
    assert result["velocity"] == BASE_STATE["velocity"]

# --- Extra positive combinations ---

def test_inlet_with_different_velocity_and_pressure():
    custom_config = {
        "boundary_conditions": [
            {
                "role": "inlet",
                "apply_to": ["velocity", "pressure"],
                "velocity": [2.0, -1.0, 0.5],
                "pressure": 150,
            }
        ]
    }
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, custom_config)
    assert result["pressure"] == 150
    assert result["velocity"] == {"vx": 2.0, "vy": -1.0, "vz": 0.5}

def test_outlet_with_custom_pressure():
    custom_config = {
        "boundary_conditions": [
            {"role": "outlet", "apply_to": ["pressure"], "pressure": 200}
        ]
    }
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, custom_config)
    assert result["pressure"] == 200
    assert result["velocity"] == BASE_STATE["velocity"]

def test_wall_with_custom_velocity():
    custom_config = {
        "boundary_conditions": [
            {"role": "wall", "apply_to": ["velocity"], "velocity": [5.0, 5.0, 5.0]}
        ]
    }
    cell = {"boundary_role": "wall"}
    result = enforce_boundary(BASE_STATE, cell, custom_config)
    assert result["velocity"] == {"vx": 5.0, "vy": 5.0, "vz": 5.0}
    assert result["pressure"] == BASE_STATE["pressure"]



