# tests/test_io_edge_cases/test_filesystem_errors.py

import os
import json
import pytest
import shutil
import tempfile
from pathlib import Path
from unittest import mock

from src.main_solver import Simulation
from src.solver.initialization import load_input_data


def test_missing_input_file_triggers_error(temp_output_dir):
    fake_path = temp_output_dir / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        load_input_data(str(fake_path))


def test_unreadable_input_file_triggers_json_error(temp_output_dir):
    bad_file = temp_output_dir / "corrupt.json"
    bad_file.write_text("{ bad json : missing quote }")
    with pytest.raises(json.JSONDecodeError):
        load_input_data(str(bad_file))


def test_simulation_fails_on_read_only_output(tmp_path):
    input_data = {
        "mesh_info": {
            "grid_shape": [2, 2, 2],
            "dx": 1.0, "dy": 1.0, "dz": 1.0
        },
        "simulation_parameters": {
            "total_time": 0.05,
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

    input_file = tmp_path / "input.json"
    with open(input_file, "w") as f:
        json.dump(input_data, f)

    output_dir = tmp_path / "readonly_output"
    output_dir.mkdir()
    output_dir.chmod(0o400)  # make it read-only

    try:
        with pytest.raises(PermissionError):
            sim = Simulation(str(input_file), str(output_dir))
            sim.run()
    finally:
        output_dir.chmod(0o700)  # restore permissions so pytest can clean up


def test_snapshot_write_permission_error_via_mock(tmp_path):
    input_data = {
        "mesh_info": {
            "grid_shape": [2, 2, 2],
            "dx": 1.0, "dy": 1.0, "dz": 1.0
        },
        "simulation_parameters": {
            "total_time": 0.05,
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

    input_file = tmp_path / "input.json"
    with open(input_file, "w") as f:
        json.dump(input_data, f)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Simulate a write failure when open() is called anywhere
    with mock.patch("builtins.open", side_effect=PermissionError("Mocked write failure")):
        with pytest.raises(PermissionError):
            sim = Simulation(str(input_file), str(output_dir))
            sim.run()



