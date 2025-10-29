# tests/test_input_reader.py

import json
import pytest
from src.input_reader import load_simulation_input


@pytest.fixture
def minimal_valid_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 2
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
            "total_time": 0.1,
            "output_interval": 1
        },
        "boundary_conditions": []
    }


def test_load_simulation_input_success(tmp_path, minimal_valid_config):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(minimal_valid_config))
    result = load_simulation_input(str(path))
    assert result["domain_definition"]["nx"] == 2
    assert result["fluid_properties"]["density"] == 1.0


def test_missing_required_section(tmp_path, minimal_valid_config):
    del minimal_valid_config["fluid_properties"]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(minimal_valid_config))
    with pytest.raises(KeyError) as e:
        load_simulation_input(str(path))
    assert "fluid_properties" in str(e.value)


def test_invalid_json(tmp_path):
    path = tmp_path / "input.json"
    path.write_text("{invalid: json}")
    with pytest.raises(ValueError):
        load_simulation_input(str(path))


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_simulation_input("nonexistent.json")


def test_pressure_solver_defaults(tmp_path, minimal_valid_config):
    minimal_valid_config["pressure_solver"] = {}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(minimal_valid_config))
    result = load_simulation_input(str(path))
    assert result["pressure_solver"] == {}


def test_ghost_rules_and_geometry(tmp_path, minimal_valid_config):
    minimal_valid_config["ghost_rules"] = {
        "boundary_faces": ["x_min", "x_max"],
        "default_type": "wall",
        "face_types": {"xmin": "inlet", "xmax": "outlet"}
    }
    minimal_valid_config["geometry_definition"] = {
        "geometry_mask_shape": [2, 2, 2],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(minimal_valid_config))
    result = load_simulation_input(str(path))
    assert result["ghost_rules"]["default_type"] == "wall"
    assert result["geometry_definition"]["flattening_order"] == "x-major"
