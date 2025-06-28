import pytest
import numpy as np
import tempfile
import os
import json
import random

from src.main_solver import Simulation

ALL_FACES = ["left", "right", "top", "bottom", "front", "back"]
BC_TYPES = ["velocity", "pressure", "neumann", "dirichlet"]

def random_boundary_conditions():
    bc = {}
    for face in ALL_FACES:
        if random.random() < 0.7:  # 70% chance of applying a BC to this face
            bc_name = f"bc_{face}"
            bc_type = random.choice(BC_TYPES)
            bc[bc_name] = {
                "type": bc_type,
                "faces": [face],
                "value": random.uniform(-1.0, 1.0) if "velocity" in bc_type else random.uniform(90000, 105000)
            }
    return bc

def create_randomized_sim_input(grid_shape=(4, 4, 4), total_time=0.1, dt=0.05, seed=123):
    random.seed(seed)
    nx, ny, nz = grid_shape
    input_data = {
        "mesh_info": {
            "grid_shape": list(grid_shape),
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0
        },
        "simulation_parameters": {
            "total_time": total_time,
            "time_step": dt,
            "density": 1.0,
            "kinematic_viscosity": 0.02,
            "output_frequency_steps": 1,
            "solver_type": "explicit"
        },
        "initial_conditions": {
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 100000.0
        },
        "boundary_conditions": random_boundary_conditions()
    }
    return input_data

@pytest.mark.parametrize("seed", [101, 202, 303])
def test_simulation_tolerates_random_bc(seed):
    sim_input = create_randomized_sim_input(seed=seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        output_dir = os.path.join(tmpdir, "output")

        with open(input_path, "w") as f:
            json.dump(sim_input, f)

        try:
            sim = Simulation(input_path, output_dir)
            sim.run()

            # Check that simulation created output directory and fields
            fields_dir = os.path.join(output_dir, "fields")
            assert os.path.isdir(fields_dir)
            assert any(fname.startswith("step_") for fname in os.listdir(fields_dir))

        except Exception as e:
            pytest.fail(f"Simulation failed with seed={seed}: {e}")



