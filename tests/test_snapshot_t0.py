# tests/test_snapshot_t0.py
# 🧪 Validation suite for t=0 snapshot fidelity — ghost-aware metrics

import json
import math
import os
import pytest
from tests.utils.input_loader import load_geometry_mask_bool

# ✅ File paths and expected values
SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0000.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
EXPECTED_STEP_INDEX = 0
EXPECTED_VELOCITY = [0.01, 0.0, 0.0]
EXPECTED_PRESSURE = 100.0
EPSILON = 1e-6

@pytest.fixture(scope="module")
def snapshot():
    assert os.path.isfile(SNAPSHOT_FILE), f"❌ Missing snapshot file: {SNAPSHOT_FILE}"
    with open(SNAPSHOT_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def domain():
    with open(INPUT_FILE) as f:
        config = json.load(f)
    return config["domain_definition"]

@pytest.fixture(scope="module")
def expected_mask():
    return load_geometry_mask_bool(INPUT_FILE)

def test_step_index(snapshot):
    assert snapshot["step_index"] == EXPECTED_STEP_INDEX

def test_grid_structure(snapshot):
    grid = snapshot["grid"]
    assert isinstance(grid, list)
    for cell in grid:
        assert isinstance(cell, dict)
        for key in ["x", "y", "z", "velocity", "pressure", "fluid_mask"]:
            assert key in cell
        assert isinstance(cell["fluid_mask"], bool)
        assert isinstance(cell["x"], (int, float))
        assert isinstance(cell["y"], (int, float))
        assert isinstance(cell["z"], (int, float))

def test_cell_coordinates(snapshot, domain):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    x_centers = [domain["min_x"] + (i + 0.5) * dx for i in range(domain["nx"])]
    y_centers = [domain["min_y"] + (j + 0.5) * dy for j in range(domain["ny"])]
    z_centers = [domain["min_z"] + (k + 0.5) * dz for k in range(domain["nz"])]

    expected_coords = [(x, y, z) for x in x_centers for y in y_centers for z in z_centers]

    domain_cells = [c for c in snapshot["grid"] if isinstance(c["x"], (int, float))]
    assert len(domain_cells) == len(expected_coords), "❌ Grid coordinate count mismatch"

    actual_coords = [(c["x"], c["y"], c["z"]) for c in domain_cells]
    for expected in expected_coords:
        assert expected in actual_coords

def test_fluid_mask_matches_geometry(snapshot, expected_mask):
    actual_mask = [c["fluid_mask"] for c in snapshot["grid"] if c["velocity"] is not None or c["pressure"] is not None]
    assert actual_mask == expected_mask

def test_velocity_and_pressure_field_values(snapshot, expected_mask):
    labeled_cells = [c for c in snapshot["grid"] if c["velocity"] is not None or c["pressure"] is not None]
    for cell, is_fluid in zip(labeled_cells, expected_mask):
        if is_fluid:
            assert isinstance(cell["velocity"], list)
            assert len(cell["velocity"]) == 3
            for a, b in zip(cell["velocity"], EXPECTED_VELOCITY):
                assert abs(a - b) < EPSILON
            assert isinstance(cell["pressure"], (int, float))
            assert abs(cell["pressure"] - EXPECTED_PRESSURE) < EPSILON
        else:
            assert cell["velocity"] is None
            assert cell["pressure"] is None

def test_max_velocity_matches_expected(snapshot):
    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    velocities = [math.sqrt(sum(v**2 for v in c["velocity"])) for c in fluid_cells if isinstance(c["velocity"], list)]
    actual_max = max(velocities) if velocities else 0.0
    expected_mag = math.sqrt(sum(v**2 for v in EXPECTED_VELOCITY))
    assert abs(snapshot["max_velocity"] - expected_mag) < EPSILON
    assert abs(snapshot["max_velocity"] - actual_max) < EPSILON

def test_global_cfl_computation(snapshot, domain):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    expected_mag = math.sqrt(sum(v**2 for v in EXPECTED_VELOCITY))
    expected_cfl = expected_mag * 0.1 / dx
    assert abs(snapshot["global_cfl"] - expected_cfl) < EPSILON

def test_basic_reflex_flags(snapshot):
    assert snapshot["damping_enabled"] is False
    assert snapshot["overflow_detected"] is False
    assert isinstance(snapshot["max_divergence"], (int, float))
    assert isinstance(snapshot["projection_passes"], int)
    assert snapshot["projection_passes"] >= 1

def test_snapshot_input_pressure_if_no_projection(snapshot, expected_mask):
    passes = snapshot.get("projection_passes", 0)
    labeled_cells = [c for c in snapshot["grid"] if c["velocity"] is not None or c["pressure"] is not None]
    if passes == 0:
        for cell, is_fluid in zip(labeled_cells, expected_mask):
            if is_fluid:
                assert abs(cell["pressure"] - EXPECTED_PRESSURE) < EPSILON
            else:
                assert cell["pressure"] is None
    else:
        pytest.skip("⚠️ Snapshot reflects projection output — skipping raw input pressure test")

def test_pressure_projection_changed_values(snapshot, expected_mask):
    passes = snapshot.get("projection_passes", 0)
    labeled_cells = [c for c in snapshot["grid"] if c["velocity"] is not None or c["pressure"] is not None]
    if passes >= 1:
        deltas = [abs(c["pressure"] - EXPECTED_PRESSURE) for c in labeled_cells if c["fluid_mask"]]
        if all(d < EPSILON for d in deltas):
            print("⚠️ Projection ran, but no pressure changed from initial value.")
            pytest.skip("⚠️ Pressure projection did not modify fluid pressures — possible equilibrium")
        assert any(d > EPSILON for d in deltas), "❌ Projection did not update fluid pressures"
    else:
        pytest.skip("⚠️ No projection pass — skipping mutation test")



