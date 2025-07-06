import pytest
from src.pre_process_input import pre_process_input_data

@pytest.fixture
def input_data_3x3x3():
    return {
        "domain_definition": {
            "min_x": 0.0,
            "max_x": 1.0,
            "min_y": 0.0,
            "max_y": 1.0,
            "min_z": 0.0,
            "max_z": 1.0,
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
                "type": "dirichlet",
                "faces": ["left"],
                "pressure": 120.0,
                "apply_to": ["pressure"]
            },
            "outlet": {
                "type": "dirichlet",
                "faces": ["right"],
                "pressure": 10.0,
                "apply_to": ["pressure"]
            }
        },
        "simulation_parameters": {
            "total_time": 1.0,
            "time_step": 0.1
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01
        },
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 100.0
        }
    }

def test_pre_process_output_grid_shape(input_data_3x3x3):
    result = pre_process_input_data(input_data_3x3x3)
    assert result["mesh_info"]["grid_shape"] == [3, 3, 3]
    assert result["domain_settings"]["dx"] > 0
    assert "boundary_conditions" in result["mesh_info"]
    assert "inlet" in result["mesh_info"]["boundary_conditions"]
    assert "cell_indices" in result["mesh_info"]["boundary_conditions"]["inlet"]
    assert "ghost_indices" in result["mesh_info"]["boundary_conditions"]["inlet"]

def test_zero_extent_mesh_z_raises():
    flat_input = {
        "domain_definition": {
            "min_x": 0.0,
            "max_x": 1.0,
            "min_y": 0.0,
            "max_y": 1.0,
            "min_z": 0.0,
            "max_z": 0.0,  # Zero extent along z
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
        # No boundary_conditions — should raise
    }

    with pytest.raises(ValueError, match="No boundary_conditions found in mesh_info"):
        pre_process_input_data(flat_input)



