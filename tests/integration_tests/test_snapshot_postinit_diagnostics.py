import os
import json
import pytest
from tests.test_helpers import decode_geometry_mask

SNAPSHOT_FILE = "data/testing-input-output/navier_stokes_output/fluid_simulation_input_step_0002.json"
INPUT_FILE = "data/testing-input-output/fluid_simulation_input.json"
EPSILON = 1e-6

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
    actual_fluid = sum(1 for cell in domain_cells if cell["fluid_mask"])
    expected_fluid = sum(expected_mask)
    assert actual_fluid == expected_fluid



