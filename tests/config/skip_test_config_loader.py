# tests/config/test_config_loader.py
# ✅ Validation suite for src/config/config_loader.py

import os
import json
import tempfile
import pytest
from src.config.config_loader import load_simulation_config

def write_temp_json(data):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    json.dump(data, temp)
    temp.close()
    return temp.name

def write_malformed_json():
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    temp.write("{ invalid json ")
    temp.close()
    return temp.name

def test_load_simulation_config_merges_domain_and_ghost_rules():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }
    ghost_rules = {
        "boundary_faces": ["x_min", "x_max"],
        "default_type": "wall",
        "face_types": {
            "x_min": "inlet",
            "x_max": "outlet"
        }
    }

    domain_path = write_temp_json(domain)
    ghost_path = write_temp_json(ghost_rules)

    config = load_simulation_config(domain_path, ghost_path, step_index=42)

    assert "domain_definition" in config
    assert "ghost_rules" in config
    assert config["step_index"] == 42

    ghost = config["ghost_rules"]
    assert ghost["default_type"] == "wall"
    assert ghost["face_types"]["xmin"] == "inlet"
    assert ghost["face_types"]["xmax"] == "outlet"

    os.remove(domain_path)
    os.remove(ghost_path)

def test_load_simulation_config_missing_domain_file():
    ghost_path = write_temp_json({"face_types": {}, "default_type": "wall", "boundary_faces": []})
    with pytest.raises(FileNotFoundError):
        load_simulation_config("nonexistent.json", ghost_path)
    os.remove(ghost_path)

def test_load_simulation_config_missing_ghost_file():
    domain_path = write_temp_json({"min_x": 0.0, "max_x": 1.0})
    with pytest.raises(FileNotFoundError):
        load_simulation_config(domain_path, "ghost_missing.json")
    os.remove(domain_path)

def test_load_simulation_config_normalizes_face_keys():
    ghost_rules = {
        "face_types": {
            "x_min": "inlet",
            "y_max": "wall",
            "z_min": "outlet"
        },
        "default_type": "generic",
        "boundary_faces": []
    }
    domain_path = write_temp_json({"min_x": 0.0, "max_x": 1.0})
    ghost_path = write_temp_json(ghost_rules)

    config = load_simulation_config(domain_path, ghost_path)
    face_types = config["ghost_rules"]["face_types"]

    assert "xmin" in face_types and face_types["xmin"] == "inlet"
    assert "ymax" in face_types and face_types["ymax"] == "wall"
    assert "zmin" in face_types and face_types["zmin"] == "outlet"

    os.remove(domain_path)
    os.remove(ghost_path)

def test_load_simulation_config_malformed_json():
    domain_path = write_temp_json({"min_x": 0.0, "max_x": 1.0})
    ghost_path = write_malformed_json()

    with pytest.raises(json.JSONDecodeError):
        load_simulation_config(domain_path, ghost_path)

    os.remove(domain_path)
    os.remove(ghost_path)

def test_load_simulation_config_missing_required_keys():
    domain_path = write_temp_json({"min_x": 0.0, "max_x": 1.0})
    ghost_path = write_temp_json({})  # missing face_types, boundary_faces, default_type

    config = load_simulation_config(domain_path, ghost_path)
    ghost = config["ghost_rules"]

    # Should fallback to empty dict and default values
    assert isinstance(ghost.get("face_types", {}), dict)
    assert ghost.get("default_type") is None or isinstance(ghost.get("default_type"), str)
    assert isinstance(ghost.get("boundary_faces", []), list)

    os.remove(domain_path)
    os.remove(ghost_path)



