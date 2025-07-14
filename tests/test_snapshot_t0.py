# tests/test_snapshot_t0.py
# 🧪 Validation suite for t=0 snapshot fidelity — ghost-aware metrics and projection verification

import json
import math
import os
import pytest
from tests.utils.input_loader import load_geometry_mask_bool

SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0000.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
EXPECTED_STEP_INDEX = 0

def is_close(actual, expected, tolerance):
    return abs(actual - expected) <= tolerance

@pytest.fixture(scope="module")
def snapshot():
    assert os.path.isfile(SNAPSHOT_FILE), f"❌ Missing snapshot file: {SNAPSHOT_FILE}"
    with open(SNAPSHOT_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def config():
    with open(INPUT_FILE) as f:
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

def test_step_index(snapshot):
    assert snapshot["step_index"] == EXPECTED_STEP_INDEX

def test_grid_structure(snapshot):
    grid = snapshot["grid"]
    assert isinstance(grid, list)
    for cell in grid:
        assert isinstance(cell, dict)
        for key in ["x", "y", "z", "velocity", "pressure", "fluid_mask"]:
            assert key in cell

def test_grid_size_matches_mask(snapshot, expected_mask):
    assert len(snapshot["grid"]) == len(expected_mask)

def test_cell_coordinates(snapshot, domain):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]
    x_centers = [domain["min_x"] + (i + 0.5) * dx for i in range(domain["nx"])]
    y_centers = [domain["min_y"] + (j + 0.5) * dy for j in range(domain["ny"])]
    z_centers = [domain["min_z"] + (k + 0.5) * dz for k in range(domain["nz"])]
    expected_coords = [(x, y, z) for x in x_centers for y in y_centers for z in z_centers]
    actual_coords = [(c["x"], c["y"], c["z"]) for c in snapshot["grid"]]
    for expected in expected_coords:
        assert expected in actual_coords

def test_fluid_mask_matches_geometry(snapshot, expected_mask):
    actual_mask = [c["fluid_mask"] for c in snapshot["grid"]]
    assert actual_mask == expected_mask

def test_velocity_and_pressure_field_values(snapshot, expected_mask, expected_velocity, expected_pressure, tolerances):
    for cell, is_fluid in zip(snapshot["grid"], expected_mask):
        if is_fluid:
            assert isinstance(cell["velocity"], list)
            for a, b in zip(cell["velocity"], expected_velocity):
                assert is_close(a, b, tolerances["velocity"])
            assert is_close(cell["pressure"], expected_pressure, tolerances["pressure"])
        else:
            assert cell["velocity"] is None
            assert cell["pressure"] is None

def test_max_velocity_matches_expected(snapshot, expected_velocity, tolerances):
    fluid_cells = [c for c in snapshot["grid"] if c["fluid_mask"]]
    velocities = [math.sqrt(sum(v**2 for v in c["velocity"])) for c in fluid_cells if isinstance(c["velocity"], list)]
    actual_max = max(velocities) if velocities else 0.0
    expected_mag = math.sqrt(sum(v**2 for v in expected_velocity))
    assert is_close(snapshot["max_velocity"], expected_mag, tolerances["velocity"])
    assert is_close(snapshot["max_velocity"], actual_max, tolerances["velocity"])

def test_global_cfl_computation(snapshot, domain, expected_velocity, tolerances):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    expected_mag = math.sqrt(sum(v**2 for v in expected_velocity))
    expected_cfl = expected_mag * 0.1 / dx
    assert is_close(snapshot["global_cfl"], expected_cfl, tolerances["cfl"])

def test_basic_reflex_flags(snapshot):
    assert isinstance(snapshot["max_divergence"], (int, float))
    assert isinstance(snapshot["projection_passes"], int)
    assert "divergence_zero" in snapshot
    assert "projection_skipped" in snapshot

def test_pressure_projection_mutated(snapshot):
    mutated_flag = snapshot.get("pressure_mutated", None)
    assert mutated_flag in [True, False]

def test_velocity_projection_applied(snapshot):
    assert snapshot.get("velocity_projected", True) is True

def test_pressure_field_changes_if_projected(snapshot, expected_mask, expected_pressure, tolerances):
    if snapshot.get("projection_passes", 0) > 0 and not snapshot.get("projection_skipped", False):
        fluid_pressures = [
            cell["pressure"] for cell, is_fluid in zip(snapshot["grid"], expected_mask) if is_fluid
        ]
        deltas = [abs(p - expected_pressure) for p in fluid_pressures]
        assert any(d > tolerances["pressure"] for d in deltas), "❌ No meaningful pressure changes despite projection"

def test_velocity_field_changes_if_projected(snapshot, expected_mask, expected_velocity, tolerances):
    if snapshot.get("projection_passes", 0) > 0 and snapshot.get("velocity_projected", True):
        fluid_velocities = [
            cell["velocity"] for cell, is_fluid in zip(snapshot["grid"], expected_mask) if is_fluid
        ]
        for v in fluid_velocities:
            assert isinstance(v, list)
        deltas = [
            math.sqrt(sum((a - b)**2 for a, b in zip(v, expected_velocity)))
            for v in fluid_velocities
        ]
        assert any(d > tolerances["velocity"] for d in deltas), "❌ Velocity field unchanged after projection"

def test_ghost_diagnostic_fields_present(snapshot):
    ghost_diag = snapshot.get("ghost_diagnostics", {})
    assert "total" in ghost_diag
    assert "per_face" in ghost_diag
    assert "pressure_overrides" in ghost_diag
    assert "no_slip_enforced" in ghost_diag



