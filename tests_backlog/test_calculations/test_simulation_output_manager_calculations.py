import json
import numpy as np
import pytest
from pathlib import Path
from src.utils import simulation_output_manager as output_mgr

class MockSimulation:
    def __init__(self, include_coords=True, all_nonzero=True):
        self.input_data = {
            "domain_definition": {"min_x": 5.0, "max_x": 10.0} if all_nonzero else {"min_x": 0.0, "max_x": 0.0},
            "fluid_properties": {"density": 1.23, "viscosity": 0.005},
            "initial_conditions": {"velocity": [3.0, 2.0, 1.0]} if all_nonzero else [0.0, 0.0, 0.0],
            "simulation_parameters": {"total_time": 5.0, "time_step": 0.02}
        }
        self.mesh_info = {
            "grid_shape": [4, 6, 2],
            "dx": 0.5,
            "dy": 0.3,
            "dz": 0.25,
            "cell_coords": np.array([[0.125, 0.125, 0.125]]) if include_coords else None,
            "face_coords": np.array([[0.0, 0.0, 0.0]]) if include_coords else None,
            "boundary_conditions": {
                "inlet": {
                    "type": "dirichlet",
                    "face_indices": np.array([5, 6, 7]),
                    "cell_indices": np.array([1, 2, 3]),
                    "velocity": [3.0, 2.0, 1.0]
                }
            }
        }

def test_setup_creates_expected_json_files_with_nonzero_fields(tmp_path):
    sim = MockSimulation()
    output_dir = tmp_path / "case"

    output_mgr.setup_simulation_output_directory(sim, str(output_dir))

    config = json.loads((output_dir / "config.json").read_text())
    mesh = json.loads((output_dir / "mesh.json").read_text())

    assert config["domain_definition"]["min_x"] == 5.0
    assert config["fluid_properties"]["viscosity"] == 0.005
    assert config["initial_conditions"]["velocity"] == [3.0, 2.0, 1.0]
    assert config["simulation_parameters"]["time_step"] == 0.02

    assert mesh["grid_shape"] == [4, 6, 2]
    assert mesh["dx"] == 0.5
    assert mesh["dy"] == 0.3
    assert mesh["dz"] == 0.25
    assert mesh["cell_coords"] == [[0.125, 0.125, 0.125]]
    assert mesh["face_coords"] == [[0.0, 0.0, 0.0]]
    assert mesh["boundary_conditions"]["inlet"]["velocity"] == [3.0, 2.0, 1.0]

def test_output_with_missing_coords(tmp_path):
    sim = MockSimulation(include_coords=False)
    output_dir = tmp_path / "missing_coords_test"
    output_mgr.setup_simulation_output_directory(sim, str(output_dir))

    mesh = json.loads((output_dir / "mesh.json").read_text())
    assert mesh["cell_coords"] is None
    assert mesh["face_coords"] is None
    assert "inlet" in mesh["boundary_conditions"]

def test_output_directory_structure_is_idempotent(tmp_path):
    sim = MockSimulation()
    output_dir = tmp_path / "reentry_case"
    output_dir.mkdir()
    (output_dir / "fields").mkdir()

    output_mgr.setup_simulation_output_directory(sim, str(output_dir))

    assert (output_dir / "config.json").exists()
    assert (output_dir / "mesh.json").exists()
    assert (output_dir / "fields").is_dir()

def test_multiple_boundary_conditions_are_serialized(tmp_path):
    sim = MockSimulation()
    sim.mesh_info["boundary_conditions"]["outlet"] = {
        "type": "neumann",
        "face_indices": np.array([9, 10]),
        "cell_indices": np.array([99, 100]),
        "pressure": 1.0
    }

    output_dir = tmp_path / "multi_bc"
    output_mgr.setup_simulation_output_directory(sim, str(output_dir))

    mesh = json.loads((output_dir / "mesh.json").read_text())
    assert "inlet" in mesh["boundary_conditions"]
    assert "outlet" in mesh["boundary_conditions"]
    assert mesh["boundary_conditions"]["outlet"]["pressure"] == 1.0
    assert mesh["boundary_conditions"]["outlet"]["face_indices"] == [9, 10]
    assert mesh["boundary_conditions"]["outlet"]["cell_indices"] == [99, 100]



