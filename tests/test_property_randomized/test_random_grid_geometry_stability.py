import pytest
import numpy as np
import tempfile
import os
import json
import random

from src.main_solver import Simulation

def create_input_with_grid_shape(shape, total_time=0.1, dt=0.05):
    nx, ny, nz = shape
    return {
        "mesh_info": {
            "grid_shape": [nx, ny, nz],
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0
        },
        "simulation_parameters": {
            "total_time": total_time,
            "time_step": dt,
            "density": 1.0,
            "kinematic_viscosity": 0.01,
            "output_frequency_steps": 1,
            "solver_type": "explicit"
        },
        "initial_conditions": {
            "velocity": [0.1, 0.0, 0.0],
            "pressure": 101325.0
        },
        "boundary_conditions": {}
    }

@pytest.mark.parametrize("shape", [
    (2, 2, 2),
    (1, 10, 10),
    (10, 1, 10),
    (10, 10, 1),
    (3, 3, 12),
    (6, 6, 6),
    (4, 7, 5),
    (8, 3, 4)
])
def test_simulation_runs_on_random_grid_geometry(shape):
    input_data = create_input_with_grid_shape(shape)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_dir = os.path.join(tmpdir, "output")

        with open(input_path, "w") as f:
            json.dump(input_data, f)

        try:
            sim = Simulation(input_path, output_dir)
            sim.run()

            fields_dir = os.path.join(output_dir, "fields")
            assert os.path.isdir(fields_dir)
            assert any(f.startswith("step_") for f in os.listdir(fields_dir))

        except Exception as e:
            pytest.fail(f"Simulation failed on grid shape {shape}: {e}")



