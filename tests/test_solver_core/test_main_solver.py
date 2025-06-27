import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

import os
import tempfile
import numpy as np
import pytest
import json

from main_solver import Simulation

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
        assert sim.velocity_field.shape == (5, 5, 5, 3)
        assert sim.p.shape == (5, 5, 5)

def test_simulation_runs_and_outputs_fields(minimal_preprocessed_input):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "solver_input.json")
        output_dir = os.path.join(tmpdir, "output")
        with open(input_path, "w") as f:
            json.dump(minimal_preprocessed_input, f)

        sim = Simulation(input_path, output_dir)
        sim.run()
        fields_dir = os.path.join(output_dir, "fields")
        assert os.path.exists(os.path.join(fields_dir, "step_0000.json"))
        assert os.path.exists(os.path.join(fields_dir, "step_0002.json"))

def test_simulation_handles_missing_input():
    with pytest.raises(FileNotFoundError):
        Simulation("nonexistent.json", "output")



