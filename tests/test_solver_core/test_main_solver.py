import os
import tempfile
import shutil
import numpy as np
import pytest
import json

from src.main_solver import Simulation

@pytest.fixture
def minimal_preprocessed_input():
    return {
        "mesh_info": {
            "grid_shape": [3, 3, 3],
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0
        },
        "simulation_parameters": {
            "total_time": 0.1,
            "time_step": 0.05,
            "density": 1.0,
            "kinematic_viscosity": 0.01,
            "output_frequency_steps": 1,
            "solver_type": "explicit"
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0
        },
        "boundary_conditions": {}
    }

def test_simulation_initializes(minimal_preprocessed_input):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "solver_input.json")
        output_dir = os.path.join(tmpdir, "output")

        with open(input_path, "w") as f:
            json.dump(minimal_preprocessed_input, f)

        sim = Simulation(input_path, output_dir)

        assert sim.nx == 3
        assert sim.velocity_field.shape == (5, 5, 5, 3)
        assert np.all(sim.velocity_field == 0.0)
        assert sim.p.shape == (5, 5, 5)
        assert sim.total_time == 0.1
        assert sim.time_step == 0.05

def test_simulation_runs_and_outputs_fields(minimal_preprocessed_input):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "solver_input.json")
        output_dir = os.path.join(tmpdir, "output")

        with open(input_path, "w") as f:
            json.dump(minimal_preprocessed_input, f)

        sim = Simulation(input_path, output_dir)
        sim.run()

        fields_path = os.path.join(output_dir, "fields")
        assert os.path.isdir(fields_path)

        snapshots = sorted(os.listdir(fields_path))
        assert "step_0000.json" in snapshots
        assert "step_0002.json" in snapshots  # 2 steps for 0.1s total @0.05 timestep

def test_simulation_handles_missing_input():
    with pytest.raises(FileNotFoundError):
        Simulation("nonexistent_file.json", "some_output_dir")



