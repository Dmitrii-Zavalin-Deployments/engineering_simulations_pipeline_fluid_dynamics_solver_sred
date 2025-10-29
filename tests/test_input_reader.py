# tests/test_input_reader.py

import os
import json
import pytest
from src.input_reader import load_simulation_input

@pytest.fixture
def valid_input(tmp_path):
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 1
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.001
        },
        "initial_conditions": {
            "initial_velocity": [0.0, 0.0, 0.0],
            "initial_pressure": 101325.0
        },
        "simulation_parameters": {
            "time_step": 0.01,
            "total_time": 1.0,
            "output_interval": 0.1
        },
        "boundary_conditions": [
            {
                "apply_to": ["inlet"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": None,
                "no_slip": False
            }
        ],
        "pressure_solver": {
            "method": "jacobi",
            "tolerance": 1e-6
        },
        "ghost_rules": {
            "boundary_faces": ["inlet", "outlet"],
            "default_type": "Dirichlet",
            "face_types": {"inlet": "Dirichlet", "outlet": "Neumann"}
        },
        "geometry_definition": {
            "geometry_mask_flat": [1, 1, 0, 1],
            "geometry_mask_shape": [2, 2, 1],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        }
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config))
    return path

def test_load_simulation_input_valid(valid_input):
    config = load_simulation_input(str(valid_input))
    assert isinstance(config, dict)
    assert "domain_definition" in config
    assert config["fluid_properties"]["density"] == 1.0
    assert config["initial_conditions"]["initial_pressure"] == 101325.0
    assert config["geometry_definition"]["geometry_mask_shape"] == [2, 2, 1]

def test_missing_file_raises(tmp_path):
    missing_path = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError) as e:
        load_simulation_input(str(missing_path))
    assert "Input file not found" in str(e.value)

def test_malformed_json_raises(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{ invalid json }")
    with pytest.raises(ValueError) as e:
        load_simulation_input(str(bad_path))
    assert "Failed to parse JSON" in str(e.value)

def test_missing_required_section_raises(tmp_path):
    incomplete = {
        "domain_definition": {},
        "fluid_properties": {},
        "initial_conditions": {},
        "simulation_parameters": {}
        # boundary_conditions missing
    }
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps(incomplete))
    with pytest.raises(KeyError) as e:
        load_simulation_input(str(path))
    assert "Missing required section" in str(e.value)

def test_optional_sections_are_safe(tmp_path):
    minimal = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "fluid_properties": {},
        "initial_conditions": {},
        "simulation_parameters": {},
        "boundary_conditions": []
        # no pressure_solver, ghost_rules, geometry_definition
    }
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(minimal))
    config = load_simulation_input(str(path))
    assert "pressure_solver" not in config or isinstance(config.get("pressure_solver"), dict)
    assert "ghost_rules" not in config or isinstance(config.get("ghost_rules"), dict)
    assert "geometry_definition" not in config or config.get("geometry_definition") is None



