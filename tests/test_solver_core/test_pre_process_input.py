import pytest
from src.pre_process_input import pre_process_input_data

def mock_boundary_faces_cube():
    return [
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
        },
        {
            "face_id": "bottom",
            "nodes": {
                "n1": [0.0, 0.0, 0.0],
                "n2": [1.0, 0.0, 0.0],
                "n3": [1.0, 0.0, 1.0],
                "n4": [0.0, 0.0, 1.0]
            }
        },
        {
            "face_id": "top",
            "nodes": {
                "n1": [0.0, 1.0, 0.0],
                "n2": [1.0, 1.0, 0.0],
                "n3": [1.0, 1.0, 1.0],
                "n4": [0.0, 1.0, 1.0]
            }
        },
        {
            "face_id": "front",
            "nodes": {
                "n1": [0.0, 0.0, 0.0],
                "n2": [0.0, 1.0, 0.0],
                "n3": [1.0, 1.0, 0.0],
                "n4": [1.0, 0.0, 0.0]
            }
        },
        {
            "face_id": "back",
            "nodes": {
                "n1": [0.0, 0.0, 1.0],
                "n2": [0.0, 1.0, 1.0],
                "n3": [1.0, 1.0, 1.0],
                "n4": [1.0, 0.0, 1.0]
            }
        }
    ]

@pytest.fixture
def input_data_3x3x3():
    return {
        "domain_definition": {
            "nx": 3,
            "ny": 3,
            "nz": 3
        },
        "mesh": {
            "boundary_faces": mock_boundary_faces_cube()
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

def test_pre_process_output_shape_and_spacing(input_data_3x3x3):
    result = pre_process_input_data(input_data_3x3x3)

    assert "domain_settings" in result
    assert "mesh_info" in result

    domain = result["domain_settings"]
    mesh = result["mesh_info"]

    assert domain["nx"] == 3
    assert domain["dy"] == pytest.approx(1.0 / 3)
    assert mesh["grid_shape"] == [3, 3, 3]

def test_zero_extent_handling():
    input_data = {
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

    result = pre_process_input_data(input_data)
    assert result["mesh_info"]["grid_shape"][2] == 1
    assert result["domain_settings"]["dz"] == 1.0



