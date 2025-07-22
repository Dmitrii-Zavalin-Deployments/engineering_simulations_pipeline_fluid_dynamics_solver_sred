# tests/test_snapshot_post_init.py
# 🧪 Snapshot Validation for t > 0 in Fluid Simulation

import os
import json
import math
import pytest
from tests.test_helpers import decode_geometry_mask

SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0002.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
EPSILON = 1e-6

@pytest.fixture(scope="module")
def snapshot():
    if not os.path.isfile(SNAPSHOT_FILE):
        pytest.skip(f"❌ Missing snapshot file: {SNAPSHOT_FILE}")
    with open(SNAPSHOT_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def config():
    if not os.path.isfile(INPUT_FILE):
        pytest.skip(f"❌ Missing input config: {INPUT_FILE}")
    with open(INPUT_FILE) as f:
        return json.load(f)

@pytest.fixture(scope="module")
def domain(config):
    return config["domain_definition"]

@pytest.fixture(scope="module")
def expected_mask(config):
    return decode_geometry_mask(config)

def get_domain_cells(snapshot, domain):
    return [c for c in snapshot["grid"]
        if domain["min_x"] <= c["x"] <= domain["max_x"] and
           domain["min_y"] <= c["y"] <= domain["max_y"] and
           domain["min_z"] <= c["z"] <= domain["max_z"]]

def test_structure_and_fields(snapshot, domain, expected_mask):
    grid = snapshot["grid"]
    domain_cells = get_domain_cells(snapshot, domain)

    assert isinstance(grid, list), "❌ Grid must be a list"
    assert len(domain_cells) == len(expected_mask), "❌ Domain-aligned grid size mismatch"

    for cell in domain_cells:
        for key in ["x", "y", "z", "velocity", "pressure", "fluid_mask"]:
            assert key in cell, f"❌ Missing '{key}' in cell"
        assert isinstance(cell["fluid_mask"], bool)
        assert isinstance(cell["x"], (int, float))
        assert isinstance(cell["y"], (int, float))
        assert isinstance(cell["z"], (int, float))

def test_fluid_vs_solid_field_behavior(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            assert cell["velocity"] is not None, "❌ Fluid cell must have velocity"
            assert isinstance(cell["velocity"], list) and len(cell["velocity"]) == 3
            assert all(isinstance(v, (int, float)) for v in cell["velocity"])
            assert cell["pressure"] is not None
            assert isinstance(cell["pressure"], (int, float))
        else:
            assert cell["velocity"] is None, "❌ Solid cell velocity must be null"
            assert cell["pressure"] is None, "❌ Solid cell pressure must be null"

def test_boundary_conditions(snapshot, domain):
    boundary_pressure = 100.0
    boundary_velocity = [0.0, 0.0, 0.0]

    for cell in snapshot["grid"]:
        is_boundary = (
            abs(cell["x"] - domain["min_x"]) < EPSILON or abs(cell["x"] - domain["max_x"]) < EPSILON or
            abs(cell["y"] - domain["min_y"]) < EPSILON or abs(cell["y"] - domain["max_y"]) < EPSILON or
            abs(cell["z"] - domain["min_z"]) < EPSILON or abs(cell["z"] - domain["max_z"]) < EPSILON
        )
        if is_boundary and cell["fluid_mask"]:
            assert math.isclose(cell["pressure"], boundary_pressure, abs_tol=EPSILON)
            assert all(math.isclose(v, bv, abs_tol=EPSILON) for v, bv in zip(cell["velocity"], boundary_velocity))

def test_velocity_magnitude(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and cell["velocity"]:
            magnitude = math.sqrt(sum(v**2 for v in cell["velocity"]))
            assert magnitude < 10.0, f"❌ Velocity magnitude exceeds overflow threshold: {magnitude}"
            assert not math.isnan(magnitude), "❌ Velocity magnitude is NaN"
            assert not math.isinf(magnitude), "❌ Velocity magnitude is infinite"

def test_max_velocity(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    magnitudes = []
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and cell["velocity"]:
            velocity_mag = math.sqrt(sum(v**2 for v in cell["velocity"]))
            assert velocity_mag < 15.0, f"❌ Velocity magnitude too high: {velocity_mag}"
            assert not math.isnan(velocity_mag), "❌ Velocity magnitude is NaN"
            assert not math.isinf(velocity_mag), "❌ Velocity magnitude is infinite"
            magnitudes.append(velocity_mag)

    max_computed = max(magnitudes) if magnitudes else 0.0
    assert math.isclose(snapshot["max_velocity"], max_computed, rel_tol=1e-5), "❌ max_velocity mismatch with grid data"

def test_global_cfl(snapshot, domain, config):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    u_max = snapshot["max_velocity"]
    dt = snapshot.get("adjusted_time_step", config["simulation_parameters"]["time_step"])
    cfl_expected = u_max * dt / dx
    assert abs(snapshot["global_cfl"] - cfl_expected) < 1e-5, "❌ global_cfl incorrect"
    assert snapshot["global_cfl"] <= 1.0, "❌ CFL exceeds stability threshold"

def test_divergence_and_projection(snapshot):
    assert isinstance(snapshot["max_divergence"], (int, float))
    assert snapshot["max_divergence"] >= 0.0
    assert isinstance(snapshot["projection_passes"], int)
    assert snapshot["projection_passes"] >= 1

def test_reflex_flags(snapshot):
    assert isinstance(snapshot["damping_enabled"], bool)
    assert isinstance(snapshot["overflow_detected"], bool)

def test_step_index_and_time(snapshot, config):
    step_index = snapshot["step_index"]
    dt = config["simulation_parameters"]["time_step"]
    sim_time = step_index * dt
    assert sim_time <= config["simulation_parameters"]["total_time"] + EPSILON

def test_fluid_cell_count(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    actual_fluid = sum(1 for cell in domain_cells if cell["fluid_mask"])
    expected_fluid = sum(expected_mask)
    assert actual_fluid == expected_fluid, f"❌ Fluid cell count mismatch: {actual_fluid} != {expected_fluid}"



