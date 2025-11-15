# tests/test_boundary_utils_missing_fields.py


import pytest
from src.step_2_time_stepping_loop.boundary_utils import enforce_boundary

# Base config and state
BASE_CONFIG = {
    "boundary_conditions": [
        {"role": "inlet", "apply_to": ["velocity", "pressure"], "velocity": [1.0, 0.0, 0.0], "pressure": 133},
        {"role": "outlet", "apply_to": ["pressure"], "pressure": 120},
        {"role": "wall", "apply_to": ["velocity"], "velocity": [0.0, 0.0, 0.0]},
    ]
}
BASE_STATE = {"pressure": 100.0, "velocity": {"vx": 0.5, "vy": 0.1, "vz": -0.2}}

# --- Missing field scenarios ---

def test_inlet_missing_apply_to():
    bad_config = {"boundary_conditions": [{"role": "inlet", "velocity": [1.0, 0.0, 0.0], "pressure": 133}]}
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result == BASE_STATE

def test_outlet_missing_apply_to():
    bad_config = {"boundary_conditions": [{"role": "outlet", "pressure": 120}]}
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result == BASE_STATE

def test_wall_missing_apply_to():
    bad_config = {"boundary_conditions": [{"role": "wall", "velocity": [0.0, 0.0, 0.0]}]}
    cell = {"boundary_role": "wall"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result == BASE_STATE

def test_inlet_missing_velocity_field():
    bad_config = {"boundary_conditions": [{"role": "inlet", "apply_to": ["velocity"]}]}  # velocity missing
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result["velocity"] == BASE_STATE["velocity"]

def test_outlet_missing_pressure_field():
    bad_config = {"boundary_conditions": [{"role": "outlet", "apply_to": ["pressure"]}]}  # pressure missing
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result["pressure"] == BASE_STATE["pressure"]

def test_velocity_wrong_length():
    bad_config = {"boundary_conditions": [{"role": "inlet", "apply_to": ["velocity"], "velocity": [1.0, 0.0]}]}
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result["velocity"] == BASE_STATE["velocity"]

def test_null_values_in_boundary_condition():
    bad_config = {"boundary_conditions": [{"role": "inlet", "apply_to": ["velocity", "pressure"], "velocity": None, "pressure": None}]}
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result == BASE_STATE

def test_empty_boundary_conditions_list():
    bad_config = {"boundary_conditions": []}
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    assert result == BASE_STATE

# --- Corrupted state scenarios ---

def test_state_missing_velocity_key():
    bad_state = {"pressure": 100.0}  # velocity missing
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(bad_state, cell, BASE_CONFIG)
    assert "pressure" in result

def test_state_missing_pressure_key():
    bad_state = {"velocity": {"vx": 0.1, "vy": 0.2, "vz": 0.3}}  # pressure missing
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(bad_state, cell, BASE_CONFIG)
    assert "velocity" in result
