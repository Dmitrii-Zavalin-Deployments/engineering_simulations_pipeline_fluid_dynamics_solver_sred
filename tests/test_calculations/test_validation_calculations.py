import pytest
from src.utils.validation import validate_json_schema

@pytest.fixture
def valid_schema():
    return {
        "fluid_properties": {
            "density": 1000.0,
            "viscosity": 0.003
        },
        "simulation_parameters": {
            "time_step": 0.05,
            "total_time": 10.0,
            "solver": "explicit"
        },
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
        },
        "boundary_conditions": {
            "inlet": {"type": "dirichlet"}
        },
        "initial_conditions": {
            "initial_velocity": [1.0, 0.0, -0.5],
            "initial_pressure": 101325.0
        }
    }

def test_valid_schema_passes(valid_schema):
    validate_json_schema(valid_schema)  # Should not raise

@pytest.mark.parametrize("missing_key", [
    "fluid_properties", "simulation_parameters", "mesh", "boundary_conditions", "initial_conditions"
])
def test_missing_top_level_key_raises(valid_schema, missing_key):
    del valid_schema[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        validate_json_schema(valid_schema)

def test_invalid_node_shape_raises(valid_schema):
    valid_schema["mesh"]["boundary_faces"][0]["nodes"]["n1"] = [1.0, 2.0]  # Only 2 coords
    with pytest.raises(ValueError, match="must be a 3-element list"):
        validate_json_schema(valid_schema)

def test_fluid_density_negative_raises(valid_schema):
    valid_schema["fluid_properties"]["density"] = -1.0
    with pytest.raises(ValueError, match="positive number"):
        validate_json_schema(valid_schema)

def test_viscosity_negative_raises(valid_schema):
    valid_schema["fluid_properties"]["viscosity"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        validate_json_schema(valid_schema)

def test_invalid_time_step_and_total_time(valid_schema):
    valid_schema["simulation_parameters"]["time_step"] = 0.0
    with pytest.raises(ValueError, match="time_step"):
        validate_json_schema(valid_schema)
    valid_schema["simulation_parameters"]["time_step"] = 0.1
    valid_schema["simulation_parameters"]["total_time"] = -5.0
    with pytest.raises(ValueError, match="total_time"):
        validate_json_schema(valid_schema)

def test_invalid_solver_string_raises(valid_schema):
    valid_schema["simulation_parameters"]["solver"] = "semi-implicit"
    with pytest.raises(ValueError, match="explicit.*implicit"):
        validate_json_schema(valid_schema)

def test_initial_velocity_wrong_length(valid_schema):
    valid_schema["initial_conditions"]["initial_velocity"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="3-element list"):
        validate_json_schema(valid_schema)

def test_initial_pressure_not_number(valid_schema):
    valid_schema["initial_conditions"]["initial_pressure"] = "high"
    with pytest.raises(ValueError, match="initial_pressure.*number"):
        validate_json_schema(valid_schema)



