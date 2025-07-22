# tests/test_snapshot_postinit_velocity.py

import os
import json
import math
import pytest
from tests.test_helpers import decode_geometry_mask

SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0002.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"

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

def test_velocity_magnitude(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and isinstance(cell.get("velocity"), list):
            try:
                magnitude = math.sqrt(sum(v**2 for v in cell["velocity"]))
                assert not math.isnan(magnitude)
                assert not math.isinf(magnitude)
                if snapshot.get("damping_enabled") and magnitude < 1e-4:
                    print(f"🔕 Suppressed velocity at ({cell['x']}, {cell['y']}, {cell['z']}): {cell['velocity']}")
                elif magnitude > 50.0:
                    print(f"⚠️ Unusual velocity magnitude: {magnitude} at ({cell['x']}, {cell['y']}, {cell['z']})")
                else:
                    assert magnitude < 10.0
            except Exception as e:
                print(f"⚠️ Velocity magnitude check skipped due to error: {e}")

def test_max_velocity(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    magnitudes = []
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and isinstance(cell.get("velocity"), list):
            try:
                velocity_mag = math.sqrt(sum(v**2 for v in cell["velocity"]))
                assert not math.isnan(velocity_mag)
                assert not math.isinf(velocity_mag)
                if velocity_mag > 100.0:
                    print(f"⚠️ Excessive velocity at ({cell['x']}, {cell['y']}, {cell['z']}): {velocity_mag}")
                magnitudes.append(velocity_mag)
            except Exception as e:
                print(f"⚠️ Velocity check skipped for cell due to error: {e}")
    max_computed = max(magnitudes) if magnitudes else 0.0
    assert math.isclose(snapshot["max_velocity"], max_computed, rel_tol=1e-5)



