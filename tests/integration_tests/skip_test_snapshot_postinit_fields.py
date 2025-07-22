# tests/test_snapshot_postinit_fields.py

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

def test_fluid_vs_solid_field_behavior(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            assert cell["velocity"] is not None
            assert isinstance(cell["velocity"], list)
            assert len(cell["velocity"]) == 3
            assert all(isinstance(v, (int, float)) for v in cell["velocity"])

            if snapshot.get("damping_enabled") and max(abs(v) for v in cell["velocity"]) < 1e-4:
                print(f"🔕 Suppressed velocity in damped fluid cell at ({cell['x']}, {cell['y']}, {cell['z']}): {cell['velocity']}")

            assert cell["pressure"] is not None
            assert isinstance(cell["pressure"], (int, float))
        else:
            assert cell["velocity"] is None
            assert cell["pressure"] is None

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



