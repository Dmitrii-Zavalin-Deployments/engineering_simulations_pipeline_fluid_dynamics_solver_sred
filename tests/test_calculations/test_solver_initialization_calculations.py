import numpy as np
import pytest
import tempfile
import json
import os

from src.solver import initialization as init

class DummySim:
    """Mock simulation instance for testing."""
    pass

@pytest.fixture
def sample_input_data():
    return {
        "simulation_parameters": {
            "total_time": 2.5,
            "time_step": 0.05,
            "density": 1.23,
            "kinematic_viscosity": 0.004,
            "output_frequency_steps": 20,
            "solver_type": "implicit"
        },
        "initial_conditions": {
            "velocity": [2.0, -1.0, 0.5],
            "pressure": 101325.0
        },
        "mesh_info": {
            "grid_shape": [4, 4, 4],
            "dx": 0.2,
            "dy": 0.2,
            "dz": 0.2
        },
        "boundary_conditions": {
            "inlet": {"type": "dirichlet", "velocity": [1.0, 0, 0]}
        }
    }

def test_initialize_simulation_parameters_assigns_values(sample_input_data):
    sim = DummySim()
    init.initialize_simulation_parameters(sim, sample_input_data)

    assert sim.total_time == 2.5
    assert sim.time_step == 0.05
    assert sim.rho == 1.23
    assert sim.nu == 0.004
    assert sim.output_frequency_steps == 20
    assert sim.solver_type == "implicit"
    assert sim.initial_velocity == [2.0, -1.0, 0.5]
    assert sim.initial_pressure == 101325.0

def test_initialize_grid_assigns_mesh_info(sample_input_data):
    sim = DummySim()
    init.initialize_grid(sim, sample_input_data)

    assert (sim.nx, sim.ny, sim.nz) == (4, 4, 4)
    assert sim.dx == 0.2
    assert sim.dy == 0.2
    assert sim.dz == 0.2

    assert "grid_shape" in sim.mesh_info
    assert "boundary_conditions" in sim.mesh_info
    assert sim.mesh_info["boundary_conditions"]["inlet"]["type"] == "dirichlet"

def test_initialize_fields_creates_correct_arrays(sample_input_data):
    sim = DummySim()
    init.initialize_simulation_parameters(sim, sample_input_data)
    init.initialize_grid(sim, sample_input_data)
    init.initialize_fields(sim, sample_input_data)

    shape = (6, 6, 6)  # (4+2, 4+2, 4+2) for ghost padding
    assert sim.u.shape == shape
    assert sim.v.shape == shape
    assert sim.w.shape == shape
    assert sim.p.shape == shape

    assert np.allclose(sim.u, 2.0)
    assert np.allclose(sim.v, -1.0)
    assert np.allclose(sim.w, 0.5)
    assert np.allclose(sim.p, 101325.0)

def test_load_input_data_reads_json(tmp_path, sample_input_data):
    path = tmp_path / "test_input.json"
    with open(path, "w") as f:
        json.dump(sample_input_data, f)

    loaded = init.load_input_data(str(path))
    assert loaded["simulation_parameters"]["total_time"] == 2.5
    assert loaded["initial_conditions"]["velocity"] == [2.0, -1.0, 0.5]

def test_load_input_data_raises_on_bad_path():
    with pytest.raises(FileNotFoundError):
        init.load_input_data("nonexistent.json")



