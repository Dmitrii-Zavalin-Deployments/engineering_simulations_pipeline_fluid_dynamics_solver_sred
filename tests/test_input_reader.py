# ✅ Unit Test Suite — Input Reader
# 📄 Full Path: tests/test_input_reader.py

import pytest
import os
import json
from tempfile import TemporaryDirectory
from src.input_reader import load_simulation_input

def build_valid_input():
    return {
        "domain_definition": {"nx": 10, "ny": 10, "nz": 10},
        "fluid_properties": {"density": 1.0, "viscosity": 0.01},
        "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0], "initial_pressure": 0.0},
        "simulation_parameters": {"output_interval": 5},
        "boundary_conditions": {
            "apply_to": ["x-min", "x-max"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 0.0,
            "no_slip": True
        },
        "pressure_solver": {"method": "gauss-seidel", "tolerance": 1e-5}
    }

def test_load_valid_input(tmp_path):
    path = tmp_path / "input.json"
    with open(path, "w") as f:
        json.dump(build_valid_input(), f)
    result = load_simulation_input(str(path))
    assert isinstance(result, dict)
    assert "domain_definition" in result
    assert result["domain_definition"]["nx"] == 10

def test_missing_file_raises():
    with pytest.raises(FileNotFoundError) as e:
        load_simulation_input("missing_file.json")
    assert "Input file not found" in str(e.value)

def test_invalid_json_raises():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "broken.json")
        with open(path, "w") as f:
            f.write("{invalid: true")  # malformed JSON
        with pytest.raises(ValueError) as e:
            load_simulation_input(path)
        assert "Failed to parse JSON" in str(e.value)

@pytest.mark.parametrize("missing_section", [
    "domain_definition",
    "fluid_properties",
    "initial_conditions",
    "simulation_parameters",
    "boundary_conditions"
])
def test_missing_required_section_raises(missing_section):
    input_data = build_valid_input()
    del input_data[missing_section]
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "input.json")
        with open(path, "w") as f:
            json.dump(input_data, f)
        with pytest.raises(KeyError) as e:
            load_simulation_input(path)
        assert f"Missing required section: {missing_section}" in str(e.value)



