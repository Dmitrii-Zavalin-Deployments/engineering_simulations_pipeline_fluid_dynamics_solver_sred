import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..", "src")))

import pytest
from pre_process_input import pre_process_input_data

@pytest.fixture
def input_data_3x3x3():
    return {
        "domain_definition": {
            "nx": 3,
            "ny": 3,
            "nz": 3
        },
        "mesh": {
            "boundary_faces": [
                {
                    "face_id": "left",
                    "nodes": {
                        "n1": [0.0, 0.0, 0.0],
                        "n2": [0.0, 1.0, 0.0],
                        "n3": [0.0, 1.0, 1.0],
                        "n4": [0.0, 0.0, 1.0]
                    }
                },
                {
                    "face_id": "right",
                    "nodes": {
                        "n1": [1.0, 0.0, 0.0],
                        "n2": [1.0, 1.0, 0.0],
                        "n3": [1.0, 1.0, 1.0],
                        "n4": [1.0, 0.0, 1.0]
                    }
                }
            ]
        },
        "boundary_conditions": {
            "inlet": {
                "type": "velocity",
                "faces": ["left"]
            }
        },
        "simulation_parameters": {
            "total_time": 1.0,
            "time_step": 0.1
        },
        "fluid_properties": {
            "density": 1.0,
            "kinematic_viscosity": 0.01
        },
        "initial_conditions": {
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 0.0
        }
    }

def test_pre_process_output_grid_shape(input_data_3x3x3):
    result = pre_process_input_data(input_data_3x3x3)
    assert result["mesh_info"]["grid_shape"] == [3, 3, 3]
    assert result["domain_settings"]["dx"] > 0

def test_zero_extent_mesh_z():
    flat_input = {
        "domain_definition": {
            "nx": 3,
            "ny": 3,
            "nz": 3
        },
        "mesh": {
            "boundary_faces": [
                {
                    "face_id": "flat",
                    "nodes": {
                        "n1": [0.0, 0.0, 0.0],
                        "n2": [1.0, 0.0, 0.0],
                        "n3": [1.0, 1.0, 0.0],
                        "n4": [0.0, 1.0, 0.0]
                    }
                }
            ]
        }
    }

    result = pre_process_input_data(flat_input)
    assert result["mesh_info"]["grid_shape"][2] == 1
    assert result["domain_settings"]["dz"] == 1.0



