import os
import json
import pytest
from tests.test_helpers import decode_geometry_mask

SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0002.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
EPSILON = 1e-6

def test_velocity_magnitude(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and cell["velocity"]:
            magnitude = math.sqrt(sum(v**2 for v in cell["velocity"]))
            assert magnitude < 10.0
            assert not math.isnan(magnitude)
            assert not math.isinf(magnitude)

def test_max_velocity(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    magnitudes = []
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid and cell["velocity"]:
            velocity_mag = math.sqrt(sum(v**2 for v in cell["velocity"]))
            assert velocity_mag < 15.0
            assert not math.isnan(velocity_mag)
            assert not math.isinf(velocity_mag)
            magnitudes.append(velocity_mag)
    max_computed = max(magnitudes) if magnitudes else 0.0
    assert math.isclose(snapshot["max_velocity"], max_computed, rel_tol=1e-5)



