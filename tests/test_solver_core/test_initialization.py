import os
import pytest
import tempfile
import json
import numpy as np

from types import SimpleNamespace
from src.solver import initialization

@pytest.fixture
def minimal_input_data():
    return {
        "simulation_parameters": {
            "total_time": 2.0,
            "time_step": 0.05,
            "density": 1.2,
            "kinematic_viscosity": 0.03,
            "output_frequency_steps": 20,
            "solver_type": "explicit"
        },
        "initial_conditions": {
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 101325
        },
        "mesh_info": {
            "grid_shape": [10, 5, 2],
            "dx": 0.1,
            "dy": 0.1,
            "dz": 0.1
        },
        "boundary_conditions": {}
    }

def test_load_input_data_valid_file(minimal_input_data):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        json.dump(minimal_input_data, temp_file)
        temp_file_path = temp_file.name

    loaded = initialization.load_input_data(temp_file_path)
    assert loaded["simulation_parameters"]["total_time"] == 2.0
    os.remove(temp_file_path)

def test_load_input_data_missing_file():
    with pytest.raises(FileNotFoundError):
        initialization.load_input_data("non_existent_file.json")

def test_initialize_simulation_parameters_sets_defaults_and_overrides(minimal_input_data):
    sim = SimpleNamespace()
    initialization.initialize_simulation_parameters(sim, minimal_input_data)

    assert sim.total_time == 2.0
    assert sim.time_step == 0.05
    assert sim.rho == 1.2
    assert sim.nu == 0.03
    assert sim.output_frequency_steps == 20
    assert sim.solver_type == "explicit"
    assert sim.initial_velocity == [1.0, 0.0, 0.0]
    assert sim.initial_pressure == 101325

def test_initialize_grid_uses_mesh_info(minimal_input_data):
    sim = SimpleNamespace()
    initialization.initialize_grid(sim, minimal_input_data)

    assert sim.nx == 10
    assert sim.ny == 5
    assert sim.nz == 2
    assert sim.dx == 0.1
    assert sim.dy == 0.1
    assert sim.dz == 0.1
    assert sim.mesh_info["grid_shape"] == [10, 5, 2]
    assert sim.mesh_info["boundary_conditions"] == {}

def test_initialize_fields_creates_correct_shapes(minimal_input_data):
    sim = SimpleNamespace()
    initialization.initialize_simulation_parameters(sim, minimal_input_data)
    initialization.initialize_grid(sim, minimal_input_data)
    initialization.initialize_fields(sim, minimal_input_data)

    expected_shape = (sim.nx + 2, sim.ny + 2, sim.nz + 2)
    assert sim.u.shape == expected_shape
    assert sim.v.shape == expected_shape
    assert sim.w.shape == expected_shape
    assert sim.p.shape == expected_shape
    assert np.all(sim.u == 1.0)
    assert np.all(sim.p == 101325)



