# tests/test_input_reader.py

import os
import json
import pytest
from src.input_reader import load_simulation_input

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
        "output_interval": 20
    },
    "boundary_conditions": [
        {
            "faces": [1, 2, 3, 4, 5, 6],
            "type": "dirichlet",
            "apply_to": ["pressure", "velocity"],
            "pressure": 100.0,
            "velocity": [0.0, 0.0, 0.0],
            "no_slip": True
        }
    ]
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
    assert result["simulation_parameters"]["output_interval"] == 20
    assert len(result["boundary_conditions"]) == 1

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
    "boundary_conditions"
])
def test_missing_required_section(tmp_path, missing_key):
    bad_input = VALID_INPUT.copy()
    bad_input.pop(missing_key)
    file = tmp_path / f"missing_{missing_key}.json"
    file.write_text(json.dumps(bad_input))
    with pytest.raises(KeyError) as exc:
        load_simulation_input(str(file))
    assert missing_key in str(exc.value)



