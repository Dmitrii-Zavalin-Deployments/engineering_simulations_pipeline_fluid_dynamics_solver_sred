# tests/test_input_reader.py
# 🧪 Validates simulation input reading, section presence, JSON parsing, and metadata logging

import os
import json
import tempfile
import pytest
from src.input_reader import load_simulation_input

def build_valid_input(overrides=None):
    base = {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1},
        "fluid_properties": {"viscosity": 0.5},
        "initial_conditions": {"velocity": [1.0, 0.0, 0.0], "pressure": 5.0},
        "simulation_parameters": {"time_step": 0.1, "output_interval": 10},
        "boundary_conditions": {
            "apply_to": ["velocity", "pressure"],
            "velocity": [0.0, 0.0, 0.0],
            "pressure": 99.0,
            "no_slip": True
        },
        "pressure_solver": {"method": "jacobi", "tolerance": 1e-5}
    }
    return {**base, **(overrides or {})}

def write_json_file(folder, name, data):
    path = os.path.join(folder, name)
    with open(path, "w") as f:
        json.dump(data, f)
    return path

def test_load_simulation_input_reads_file_success(capsys):
    with tempfile.TemporaryDirectory() as folder:
        path = write_json_file(folder, "input.json", build_valid_input())
        result = load_simulation_input(path)
        assert isinstance(result, dict)
        assert "domain_definition" in result
        out = capsys.readouterr().out
        assert "Domain resolution" in out
        assert "Pressure Solver" in out
        assert "Boundary Conditions" in out

def test_load_simulation_input_missing_file_raises():
    with tempfile.TemporaryDirectory() as folder:
        bad_path = os.path.join(folder, "missing.json")
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            load_simulation_input(bad_path)

def test_load_simulation_input_invalid_json_raises():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "bad.json")
        with open(path, "w") as f:
            f.write("{invalid: json}")
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            load_simulation_input(path)

@pytest.mark.parametrize("missing_section", [
    "domain_definition", "fluid_properties", "initial_conditions",
    "simulation_parameters", "boundary_conditions"
])
def test_load_simulation_input_missing_section_raises(missing_section):
    with tempfile.TemporaryDirectory() as folder:
        data = build_valid_input()
        del data[missing_section]
        path = write_json_file(folder, "input.json", data)
        with pytest.raises(KeyError, match="Missing required section"):
            load_simulation_input(path)

def test_load_simulation_input_handles_missing_pressure_solver_defaults(capsys):
    with tempfile.TemporaryDirectory() as folder:
        data = build_valid_input()
        del data["pressure_solver"]
        path = write_json_file(folder, "input.json", data)
        result = load_simulation_input(path)
        assert "pressure_solver" not in result  # confirms default applied
        out = capsys.readouterr().out
        assert "Method: jacobi" in out
        assert "Tolerance: 1e-06" in out

def test_load_simulation_input_output_interval_printed(capsys):
    with tempfile.TemporaryDirectory() as folder:
        data = build_valid_input()
        data["simulation_parameters"]["output_interval"] = 25
        path = write_json_file(folder, "input.json", data)
        load_simulation_input(path)
        out = capsys.readouterr().out
        assert "Output interval: 25" in out