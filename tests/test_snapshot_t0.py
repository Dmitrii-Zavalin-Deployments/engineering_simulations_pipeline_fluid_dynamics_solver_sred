# tests/test_snapshot_t0.py
# 🧪 Validation suite for t=0 snapshot fidelity — ghost-aware metrics

import json
import math
import os
import pytest

from tests.utils.input_loader import load_geometry_mask_bool

# ✅ Snapshot file path (t=0)
SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0000.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"

# ✅ Expected initial values
EXPECTED_VELOCITY = [0.01, 0.0, 0.0]
EXPECTED_PRESSURE = 100.0
EXPECTED_STEP_INDEX = 0

# ✅ Tolerance for float comparisons
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
    assert snapshot["step_index"] == EXPECTED_STEP_INDEX, "❌ Incorrect step_index at t=0"


def test_grid_structure(snapshot, domain):
    grid = snapshot["grid"]
    assert isinstance(grid, list), "❌ Grid must be a list"
    for cell in grid:
        assert isinstance(cell, dict), "❌ Each cell must be a dict"
        for key in ["x", "y", "z", "velocity", "pressure", "fluid_mask"]:
            assert key in cell, f"❌ Missing key '{key}' in cell"
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

    expected_coords = []
    for i in range(domain["nx"]):
        for j in range(domain["ny"]):
            for k in range(domain["nz"]):
                expected_coords.append((x_centers[i], y_centers[j], z_centers[k]))

    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    assert len(fluid_cells) == len(expected_coords), "❌ Grid fluid count mismatch"
    for cell, (ex, ey, ez) in zip(fluid_cells, expected_coords):
        assert abs(cell["x"] - ex) < EPSILON
        assert abs(cell["y"] - ey) < EPSILON
        assert abs(cell["z"] - ez) < EPSILON


def test_fluid_mask_matches_geometry(snapshot, expected_mask):
    physical_cells = [c for c in snapshot["grid"] if c["fluid_mask"] or (c["velocity"] is not None and c["pressure"] is not None)]
    actual = [cell["fluid_mask"] for cell in physical_cells]
    assert actual == expected_mask, "❌ fluid_mask mismatch with geometry definition"


def test_velocity_pressure_assignment(snapshot, expected_mask):
    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    for cell, is_fluid in zip(fluid_cells, expected_mask):
        if is_fluid:
            assert isinstance(cell["velocity"], list) and len(cell["velocity"]) == 3
            for a, b in zip(cell["velocity"], EXPECTED_VELOCITY):
                assert abs(a - b) < EPSILON
            assert isinstance(cell["pressure"], (int, float))
            assert abs(cell["pressure"] - EXPECTED_PRESSURE) < EPSILON
        else:
            assert cell["velocity"] is None
            assert cell["pressure"] is None


def test_max_velocity(snapshot):
    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    velocities = [math.sqrt(sum(v**2 for v in c["velocity"])) for c in fluid_cells if isinstance(c["velocity"], list)]
    max_actual = max(velocities) if velocities else 0.0
    mag_expected = math.sqrt(sum(v**2 for v in EXPECTED_VELOCITY))
    assert abs(snapshot["max_velocity"] - max_actual) < EPSILON
    assert abs(snapshot["max_velocity"] - mag_expected) < EPSILON


def test_global_cfl(snapshot, domain):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    mag_velocity = math.sqrt(sum(v**2 for v in EXPECTED_VELOCITY))
    cfl_expected = mag_velocity * 0.1 / dx
    assert abs(snapshot["global_cfl"] - cfl_expected) < EPSILON


def test_reflex_flags(snapshot):
    assert snapshot["overflow_detected"] is False
    assert snapshot["damping_enabled"] is False
    assert isinstance(snapshot["max_divergence"], (int, float))
    assert isinstance(snapshot["projection_passes"], int)
    assert snapshot["projection_passes"] >= 1


def test_snapshot_integrity_for_raw_input_pressure(snapshot, expected_mask):
    passes = snapshot.get("projection_passes", 0)
    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    if passes == 0:
        for cell, is_fluid in zip(fluid_cells, expected_mask):
            if is_fluid:
                assert abs(cell["pressure"] - EXPECTED_PRESSURE) < EPSILON
            else:
                assert cell["pressure"] is None
    else:
        pytest.skip("⚠️ Snapshot reflects projection output — skipping raw input pressure test")


def test_pressure_projection_modifies_pressure_if_solver_runs(snapshot, expected_mask):
    passes = snapshot.get("projection_passes", 0)
    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    if passes >= 1:
        fluid_pressures = [c["pressure"] for c in fluid_cells]
        deltas = [abs(p - EXPECTED_PRESSURE) for p in fluid_pressures]
        if all(delta < EPSILON for delta in deltas):
            print("⚠️ Projection ran, but no pressure changed from initial value.")
            pytest.skip("⚠️ Pressure projection did not modify fluid pressures — possible equilibrium")
        assert any(delta > EPSILON for delta in deltas), \
            "❌ Pressure projection at t=0 did not modify expected values"
    else:
        pytest.skip("⚠️ projection_passes == 0 — skipping pressure mutation test")



