# tests/test_snapshot_postinit_structure.py

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

        if isinstance(cell["velocity"], list) and len(cell["velocity"]) == 3:
            assert all(isinstance(v, (int, float)) for v in cell["velocity"])
            if snapshot.get("damping_enabled") and max(abs(v) for v in cell["velocity"]) < 1e-4:
                print(f"🔕 Suppressed velocity detected at ({cell['x']}, {cell['y']}, {cell['z']}): {cell['velocity']}")



