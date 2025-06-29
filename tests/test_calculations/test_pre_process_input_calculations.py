import pytest
from src.pre_process_input import pre_process_input_data

@pytest.fixture
def minimal_input():
    return {
        "domain_definition": {},
        "fluid_properties": {"density": 997.0, "viscosity": 0.001},
        "simulation_parameters": {"total_time": 1.0, "time_step": 0.1},
        "initial_conditions": {"initial_velocity": [1.0, 0.0, 0.0], "initial_pressure": 101325.0},
        "boundary_conditions": {"inlet": {"type": "dirichlet", "velocity": [1.0, 0.0, 0.0]}},
        "mesh": {
            "boundary_faces": [
                {
                    "face_id": "f1",
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

def test_grid_inference_produces_expected_shape(minimal_input):
    output = pre_process_input_data(minimal_input)
    grid = output["domain_settings"]
    mesh_info = output["mesh_info"]

    assert grid["nx"] == 1
    assert grid["ny"] == 1
    assert grid["nz"] == 1  # All z coords are 0 → collapsed → fallback to 1
    assert grid["dx"] == pytest.approx(1.0)
    assert mesh_info["grid_shape"] == [1, 1, 1]

def test_user_defined_nx_is_respected(minimal_input):
    minimal_input["domain_definition"]["nx"] = 4
    output = pre_process_input_data(minimal_input)

    assert output["domain_settings"]["nx"] == 4
    assert output["domain_settings"]["dx"] == pytest.approx(0.25)

def test_zero_extent_forces_unit_grid(minimal_input):
    # Collapse all x and y
    for face in minimal_input["mesh"]["boundary_faces"]:
        for node in face["nodes"]:
            face["nodes"][node] = [0.0, 0.0, 0.0]

    output = pre_process_input_data(minimal_input)
    grid = output["domain_settings"]

    assert grid["nx"] == 1
    assert grid["ny"] == 1
    assert grid["nz"] == 1
    assert grid["dx"] == 1.0
    assert grid["dy"] == 1.0
    assert grid["dz"] == 1.0

def test_preprocessed_output_structure(minimal_input):
    output = pre_process_input_data(minimal_input)

    assert "domain_settings" in output
    assert "fluid_properties" in output
    assert "simulation_parameters" in output
    assert "initial_conditions" in output
    assert "boundary_conditions" in output
    assert "mesh_info" in output
    assert "mesh" in output
    assert "boundary_faces" in output["mesh"]

def test_all_physical_fields_are_preserved(minimal_input):
    output = pre_process_input_data(minimal_input)

    assert output["fluid_properties"]["density"] == 997.0
    assert output["initial_conditions"]["initial_velocity"] == [1.0, 0.0, 0.0]
    assert output["initial_conditions"]["initial_pressure"] == 101325.0
    assert output["simulation_parameters"]["time_step"] == 0.1



