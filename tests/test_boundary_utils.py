# tests/test_boundary_utils.py

import pytest
from src.step_2_time_stepping_loop.boundary_utils import enforce_boundary

# Shared config (the JSON you provided)
CONFIG = {
    "domain_definition": {
        "x_min": -0.0,
        "x_max": 3.0,
        "y_min": -1.5,
        "y_max": 1.5,
        "z_min": -1.5,
        "z_max": 1.5,
        "nx": 4,
        "ny": 4,
        "nz": 4,
    },
    "fluid_properties": {"density": 1.137, "viscosity": 0.09},
    "initial_conditions": {
        "initial_velocity": [1, 0, 0],
        "initial_pressure": 133.105,
    },
    "simulation_parameters": {
        "time_step": 0.1,
        "total_time": 1.0,
        "output_interval": 2,
    },
    "boundary_conditions": [
        {
            "role": "inlet",
            "type": "dirichlet",
            "faces": [1],
            "apply_to": ["velocity", "pressure"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 133,
            "apply_faces": ["x_min"],
        },
        {
            "role": "outlet",
            "type": "neumann",
            "faces": [7],
            "apply_to": ["pressure"],
            "apply_faces": ["x_max"],
        },
        {
            "role": "wall",
            "type": "dirichlet",
            "faces": [6],
            "apply_to": ["velocity"],
            "no_slip": True,
            "velocity": [0.0, 0.0, 0.0],
            "apply_faces": ["wall"],
        },
    ],
    "geometry_definition": {
        "geometry_mask_flat": [-1] * 64,
        "geometry_mask_shape": [4, 4, 4],
        "mask_encoding": {"fluid": 1, "solid": 0, "boundary": -1},
        "flattening_order": "x-major",
    },
}

BASE_STATE = {
    "pressure": 120.0,
    "velocity": {"vx": 0.5, "vy": 0.1, "vz": -0.2},
}

# --- Standard cases ---

def test_no_boundary_role_returns_unchanged():
    cell = {"boundary_role": None}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result == BASE_STATE

def test_inlet_overrides_velocity_and_pressure():
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result["pressure"] == 133
    assert result["velocity"] == {"vx": 1.0, "vy": 0.0, "vz": 0.0}

def test_outlet_overrides_pressure_only():
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    # Pressure overridden, velocity unchanged
    assert result["pressure"] == CONFIG["boundary_conditions"][1].get("pressure", BASE_STATE["pressure"])
    assert result["velocity"] == BASE_STATE["velocity"]

def test_wall_overrides_velocity_only():
    cell = {"boundary_role": "wall"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result["velocity"] == {"vx": 0.0, "vy": 0.0, "vz": 0.0}
    assert result["pressure"] == BASE_STATE["pressure"]

def test_unknown_role_returns_unchanged():
    cell = {"boundary_role": "ghost"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result == BASE_STATE

def test_boundary_role_with_empty_apply_to_returns_unchanged():
    # Add a fake boundary condition with empty apply_to
    CONFIG["boundary_conditions"].append({
        "role": "test_empty",
        "apply_to": []
    })
    cell = {"boundary_role": "test_empty"}
    result = enforce_boundary(BASE_STATE, cell, CONFIG)
    assert result == BASE_STATE

# --- Edge / corrupted cases ---

def test_missing_boundary_conditions_key_in_config():
    bad_config = CONFIG.copy()
    bad_config.pop("boundary_conditions", None)
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should return unchanged because no boundary_conditions exist
    assert result == BASE_STATE

def test_boundary_condition_missing_velocity_field():
    bad_config = {
        **CONFIG,
        "boundary_conditions": [
            {"role": "inlet", "apply_to": ["velocity"]}  # velocity missing
        ]
    }
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should not crash, velocity unchanged
    assert result["velocity"] == BASE_STATE["velocity"]

def test_boundary_condition_missing_pressure_field():
    bad_config = {
        **CONFIG,
        "boundary_conditions": [
            {"role": "outlet", "apply_to": ["pressure"]}  # pressure missing
        ]
    }
    cell = {"boundary_role": "outlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should not crash, pressure unchanged
    assert result["pressure"] == BASE_STATE["pressure"]

def test_boundary_condition_velocity_wrong_length():
    bad_config = {
        **CONFIG,
        "boundary_conditions": [
            {"role": "inlet", "apply_to": ["velocity"], "velocity": [1.0, 0.0]}  # only 2 components
        ]
    }
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should not crash, velocity unchanged
    assert result["velocity"] == BASE_STATE["velocity"]

def test_boundary_condition_apply_to_missing():
    bad_config = {
        **CONFIG,
        "boundary_conditions": [
            {"role": "wall", "velocity": [0.0, 0.0, 0.0]}  # apply_to missing
        ]
    }
    cell = {"boundary_role": "wall"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should not crash, state unchanged
    assert result == BASE_STATE

def test_boundary_condition_null_values():
    bad_config = {
        **CONFIG,
        "boundary_conditions": [
            {"role": "inlet", "apply_to": ["velocity", "pressure"], "velocity": None, "pressure": None}
        ]
    }
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should not crash, state unchanged
    assert result == BASE_STATE

def test_boundary_condition_empty_list():
    bad_config = {
        **CONFIG,
        "boundary_conditions": []
    }
    cell = {"boundary_role": "inlet"}
    result = enforce_boundary(BASE_STATE, cell, bad_config)
    # Should not crash, state unchanged
    assert result == BASE_STATE

def test_state_missing_velocity_key():
    cell = {"boundary_role": "inlet"}
    bad_state = {"pressure": 100.0}  # velocity missing
    result = enforce_boundary(bad_state, cell, CONFIG)
    # Should not crash, pressure overridden but velocity remains missing
    assert "pressure" in result

def test_state_missing_pressure_key():
    cell = {"boundary_role": "inlet"}
    bad_state = {"velocity": {"vx": 0.1, "vy": 0.2, "vz": 0.3}}  # pressure missing
    result = enforce_boundary(bad_state, cell, CONFIG)
    # Should not crash, velocity overridden, pressure remains missing or unchanged
    assert "velocity" in result



