# tests/integration_tests/snapshot_t0_shared.py
# 🧪 Shared fixtures and utilities for t=0 snapshot validation

import os
import json
import math
import pytest

INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0000.json"
EXPECTED_STEP_INDEX = 0

def load_geometry_mask_bool(path):
    with open(path, "r") as f:
        data = json.load(f)
    raw = data["geometry_definition"]["geometry_mask_flat"]
    fluid_code = data["geometry_definition"]["mask_encoding"]["fluid"]
    return [v == fluid_code for v in raw]

def get_domain_cells(snapshot, domain):
    return [c for c in snapshot["grid"]
        if domain["min_x"] <= c["x"] <= domain["max_x"] and
           domain["min_y"] <= c["y"] <= domain["max_y"] and
           domain["min_z"] <= c["z"] <= domain["max_z"]]

def is_close(actual, expected, tolerance):
    return abs(actual - expected) <= tolerance

def get_effective_timestep(snapshot, config):
    return snapshot.get("adjusted_time_step", config["simulation_parameters"]["time_step"])

def get_snapshot_flags(snapshot):
    return {
        "step_index": snapshot.get("step_index"),
        "projection_passes": snapshot.get("projection_passes", 1),
        "damping_enabled": snapshot.get("damping_enabled", False),
        "overflow_detected": snapshot.get("overflow_detected", False)
    }

@pytest.fixture(scope="module")
def config():
    if os.path.isfile(INPUT_FILE):
        with open(INPUT_FILE) as f:
            return json.load(f)
    print("⚠️ Fallback config used: disk file not found")
    return {
        "geometry_path": "data/geometry.json",
        "fluid_mask_path": "data/mask.json",
        "nx": 3,
        "ny": 1,
        "nz": 1,
        "min_x": 0.0,
        "max_x": 3.0,
        "min_y": 0.0,
        "max_y": 1.0,
        "min_z": 0.0,
        "max_z": 1.0,
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        },
        "initial_conditions": {
            "initial_velocity": [0.1, 0.0, 0.0],
            "initial_pressure": 102.156
        },
        "simulation_parameters": {
            "time_step": 0.1
        }
    }

@pytest.fixture(scope="module")
def snapshot():
    if not os.path.isfile(SNAPSHOT_FILE):
        pytest.skip(f"❌ Missing snapshot file: {SNAPSHOT_FILE}")
    with open(SNAPSHOT_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def domain(config):
    return config["domain_definition"]

@pytest.fixture(scope="module")
def expected_velocity(config):
    return config["initial_conditions"]["initial_velocity"]

@pytest.fixture(scope="module")
def expected_pressure(config):
    return config["initial_conditions"]["initial_pressure"]

@pytest.fixture(scope="module")
def expected_mask():
    return load_geometry_mask_bool(INPUT_FILE)

@pytest.fixture(scope="module")
def tolerances(expected_velocity, expected_pressure):
    v_tol = max(abs(v) for v in expected_velocity if v != 0) * 0.01
    p_tol = max(abs(expected_pressure), 1e-6) * 0.01
    return {
        "velocity": v_tol,
        "pressure": p_tol,
        "cfl": v_tol
    }

@pytest.fixture(scope="module")
def snapshot_flags(snapshot):
    return get_snapshot_flags(snapshot)



