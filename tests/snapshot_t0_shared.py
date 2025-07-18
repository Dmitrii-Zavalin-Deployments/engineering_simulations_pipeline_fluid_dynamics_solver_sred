# tests/snapshot_t0_shared.py
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

@pytest.fixture(scope="module")
def config():
    if not os.path.isfile(INPUT_FILE):
        pytest.skip(f"❌ Missing input config: {INPUT_FILE}")
    with open(INPUT_FILE) as f:
        return json.load(f)

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



