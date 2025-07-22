# tests/test_snapshot_postinit_diagnostics.py

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

def test_global_cfl(snapshot, domain, config):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    u_max = snapshot["max_velocity"]
    dt = snapshot.get("adjusted_time_step", config["simulation_parameters"]["time_step"])
    expected_cfl = u_max * dt / dx
    assert abs(snapshot["global_cfl"] - expected_cfl) < 1e-5
    assert snapshot["global_cfl"] <= 1.0

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
    actual_fluid = 0
    expected_fluid = sum(expected_mask)

    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and cell["fluid_mask"]:
            actual_fluid += 1

            # 🔕 Damping-aware velocity suppression trace
            if snapshot.get("damping_enabled"):
                v = cell.get("velocity")
                if isinstance(v, list) and len(v) == 3 and max(abs(c) for c in v) < 1e-4:
                    print(f"🔕 Suppressed velocity at ({cell['x']}, {cell['y']}, {cell['z']}): {v}")

    assert actual_fluid == expected_fluid, f"❌ Fluid cell count mismatch: {actual_fluid} != {expected_fluid}"



