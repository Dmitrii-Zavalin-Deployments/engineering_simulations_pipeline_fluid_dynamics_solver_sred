# tests/test_input_reader.py

import os
import json
import pytest
from input_reader import load_simulation_input

VALID_INPUT = {
    "domain_definition": {
        "min_x": 0.0, "max_x": 3.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 41, "ny": 41, "nz": 11
    },
    "fluid_properties": {
        "density": 1.0, "viscosity": 0.01
    },
    "initial_conditions": {
        "initial_velocity": [0.01, 0.0, 0.0],
        "initial_pressure": 100.0
    },
    "simulation_parameters": {
        "time_step": 0.1, "total_time": 1.0,
        "solver": "explicit", "output_frequency_steps": 20
    },
    "boundary_conditions": [
        {"label": "inlet", "faces": [1], "type": "dirichlet", "apply_to": ["pressure"], "pressure": 100.0}
    ],
    "mesh": {
        "boundary_faces": [
            {"face_id": 1, "nodes": {"n1": [0,0,0], "n2": [0,1,0], "n3": [0,0,1], "n4": [0,1,1]}}
        ]
    }
}

@pytest.fixture
def tmp_valid_input(tmp_path):
    file = tmp_path / "valid_input.json"
    file.write_text(json.dumps(VALID_INPUT))
    return str(file)

def test_valid_input_loads_correctly(tmp_valid_input):
    result = load_simulation_input(tmp_valid_input)
    assert result["domain_definition"]["nx"] == 41
    assert result["fluid_properties"]["density"] == 1.0
    assert result["initial_conditions"]["initial_velocity"] == [0.01, 0.0, 0.0]
    assert result["simulation_parameters"]["solver"] == "explicit"
    assert len(result["boundary_conditions"]) == 1
    assert "mesh" in result

def test_missing_input_file():
    with pytest.raises(FileNotFoundError):
        load_simulation_input("nonexistent_file.json")

def test_invalid_json_content(tmp_path):
    bad_file = tmp_path / "bad_input.json"
    bad_file.write_text("{ invalid_json: }")
    with pytest.raises(ValueError):
        load_simulation_input(str(bad_file))

@pytest.mark.parametrize("missing_key", [
    "domain_definition",
    "fluid_properties",
    "initial_conditions",
    "simulation_parameters",
    "boundary_conditions",
    "mesh"
])
def test_missing_required_section(tmp_path, missing_key):
    bad_input = VALID_INPUT.copy()
    bad_input.pop(missing_key)
    file = tmp_path / f"missing_{missing_key}.json"
    file.write_text(json.dumps(bad_input))
    with pytest.raises(KeyError) as exc:
        load_simulation_input(str(file))
    assert missing_key in str(exc.value)

def test_optional_solver_defaults(tmp_path):
    no_solver = VALID_INPUT.copy()
    no_solver["simulation_parameters"] = {
        "time_step": 0.1, "total_time": 1.0, "output_frequency_steps": 20
    }
    file = tmp_path / "missing_solver.json"
    file.write_text(json.dumps(no_solver))
    result = load_simulation_input(str(file))
    assert result["simulation_parameters"].get("solver", "unknown") == "unknown"



