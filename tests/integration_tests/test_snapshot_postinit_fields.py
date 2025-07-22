import os
import json
import pytest
from tests.test_helpers import decode_geometry_mask

SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0002.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
EPSILON = 1e-6

def test_fluid_vs_solid_field_behavior(snapshot, domain, expected_mask):
    domain_cells = get_domain_cells(snapshot, domain)
    for cell, is_fluid in zip(domain_cells, expected_mask):
        if is_fluid:
            assert cell["velocity"] is not None
            assert isinstance(cell["velocity"], list) and len(cell["velocity"]) == 3
            assert all(isinstance(v, (int, float)) for v in cell["velocity"])
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



