# tests/test_snapshot_t0.py

import json
import math
import os
import pytest

# ✅ Snapshot file path (t=0)
SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0000.json"

# ✅ Expected initial values
EXPECTED_VELOCITY = [0.01, 0.0, 0.0]
EXPECTED_PRESSURE = 100.0
EXPECTED_STEP_INDEX = 0

# ✅ Domain details
DOMAIN = {
    "min_x": 0.0,
    "max_x": 3.0,
    "min_y": 0.0,
    "max_y": 1.0,
    "min_z": 0.0,
    "max_z": 1.0,
    "nx": 3,
    "ny": 2,
    "nz": 1
}

# ✅ Geometry mask: x-major flattening
EXPECTED_MASK_FLAT = [1, 1, 0, 0, 1, 1]
EXPECTED_MASK_BOOL = [bool(v) for v in EXPECTED_MASK_FLAT]

# ✅ Tolerance for float comparisons
EPSILON = 1e-6

# ✅ Grid spacing
DX = (DOMAIN["max_x"] - DOMAIN["min_x"]) / DOMAIN["nx"]
DY = (DOMAIN["max_y"] - DOMAIN["min_y"]) / DOMAIN["ny"]
DZ = (DOMAIN["max_z"] - DOMAIN["min_z"]) / DOMAIN["nz"]


@pytest.fixture(scope="module")
def snapshot():
    assert os.path.isfile(SNAPSHOT_FILE), f"❌ Missing snapshot file: {SNAPSHOT_FILE}"
    with open(SNAPSHOT_FILE) as f:
        return json.load(f)


def test_step_index(snapshot):
    assert snapshot["step_index"] == EXPECTED_STEP_INDEX, "❌ Incorrect step_index at t=0"


def test_grid_structure(snapshot):
    grid = snapshot["grid"]
    assert isinstance(grid, list), "❌ Grid must be a list"
    assert len(grid) == DOMAIN["nx"] * DOMAIN["ny"] * DOMAIN["nz"], "❌ Grid size mismatch"
    for cell in grid:
        assert isinstance(cell, dict), "❌ Each cell must be a dict"
        for key in ["x", "y", "z", "velocity", "pressure", "fluid_mask"]:
            assert key in cell, f"❌ Missing key '{key}' in cell"
        assert isinstance(cell["fluid_mask"], bool)
        assert isinstance(cell["x"], (int, float))
        assert isinstance(cell["y"], (int, float))
        assert isinstance(cell["z"], (int, float))


def test_cell_coordinates(snapshot):
    grid = snapshot["grid"]
    x_centers = [DOMAIN["min_x"] + (i + 0.5) * DX for i in range(DOMAIN["nx"])]
    y_centers = [DOMAIN["min_y"] + (j + 0.5) * DY for j in range(DOMAIN["ny"])]
    z_centers = [DOMAIN["min_z"] + (k + 0.5) * DZ for k in range(DOMAIN["nz"])]

    expected_coords = []
    for i in range(DOMAIN["nx"]):
        for j in range(DOMAIN["ny"]):
            for k in range(DOMAIN["nz"]):
                expected_coords.append((x_centers[i], y_centers[j], z_centers[k]))

    for cell, (ex, ey, ez) in zip(grid, expected_coords):
        assert abs(cell["x"] - ex) < EPSILON
        assert abs(cell["y"] - ey) < EPSILON
        assert abs(cell["z"] - ez) < EPSILON


def test_fluid_mask_matches_geometry(snapshot):
    actual = [cell["fluid_mask"] for cell in snapshot["grid"]]
    assert actual == EXPECTED_MASK_BOOL, "❌ fluid_mask mismatch with geometry definition"


def test_velocity_pressure_assignment(snapshot):
    for cell, is_fluid in zip(snapshot["grid"], EXPECTED_MASK_BOOL):
        if is_fluid:
            assert isinstance(cell["velocity"], list) and len(cell["velocity"]) == 3
            for a, b in zip(cell["velocity"], EXPECTED_VELOCITY):
                assert abs(a - b) < EPSILON
            assert isinstance(cell["pressure"], (int, float))
            assert abs(cell["pressure"] - EXPECTED_PRESSURE) < EPSILON
        else:
            assert cell["velocity"] is None, "❌ Solid cell velocity must be null"
            assert cell["pressure"] is None, "❌ Solid cell pressure must be null"


def test_max_velocity(snapshot):
    mag_expected = math.sqrt(sum(v**2 for v in EXPECTED_VELOCITY))
    assert abs(snapshot["max_velocity"] - mag_expected) < EPSILON, "❌ Incorrect max_velocity"


def test_global_cfl(snapshot):
    cfl_expected = math.sqrt(sum(v**2 for v in EXPECTED_VELOCITY)) * 0.1 / DX
    assert abs(snapshot["global_cfl"] - cfl_expected) < EPSILON, "❌ Incorrect global_cfl"


def test_reflex_flags(snapshot):
    assert snapshot["overflow_detected"] is False, "❌ overflow_detected should be false at t=0"
    assert snapshot["damping_enabled"] is False, "❌ damping_enabled should be false at t=0"
    assert isinstance(snapshot["max_divergence"], (int, float))
    assert isinstance(snapshot["projection_passes"], int)
    assert snapshot["projection_passes"] >= 1



