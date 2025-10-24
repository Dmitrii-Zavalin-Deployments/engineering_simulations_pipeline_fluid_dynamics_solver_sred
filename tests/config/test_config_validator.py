# tests/config/test_config_validator.py
# ✅ Validation suite for src/config/config_validator.py

import pytest
from src.config.config_validator import validate_config

def valid_config():
    return {
        "domain_definition": {
            "min_x": 0, "max_x": 10,
            "min_y": 0, "max_y": 10,
            "min_z": 0, "max_z": 10
        },
        "ghost_rules": {
            "boundary_faces": ["inlet", "outlet"],
            "default_type": "wall",
            "face_types": {"inlet": "dirichlet", "outlet": "neumann"}
        },
        "boundary_conditions": [
            {
                "apply_to": ["inlet"],
                "type": "dirichlet",
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 101325,
                "no_slip": False
            },
            {
                "apply_to": ["outlet"],
                "type": "neumann"
            }
        ]
    }

def test_valid_config_passes(capsys):
    validate_config(valid_config())
    assert "[CONFIG] Validation passed" in capsys.readouterr().out

@pytest.mark.parametrize("bad_type", [None, [], "string", 42])
def test_config_must_be_dict(bad_type):
    with pytest.raises(ValueError, match="Config must be a dictionary."):
        validate_config(bad_type)

@pytest.mark.parametrize("missing_key", ["min_x", "max_y", "min_z"])
def test_missing_domain_keys_raise(missing_key):
    cfg = valid_config()
    del cfg["domain_definition"][missing_key]
    with pytest.raises(ValueError, match=f"Missing or invalid '{missing_key}'"):
        validate_config(cfg)

@pytest.mark.parametrize("bad_type", [None, "string", [], 42])
def test_domain_keys_must_be_numeric(bad_type):
    cfg = valid_config()
    cfg["domain_definition"]["min_x"] = bad_type
    with pytest.raises(ValueError, match="Missing or invalid 'min_x'"):
        validate_config(cfg)

def test_ghost_rules_missing_keys_raise():
    cfg = valid_config()
    del cfg["ghost_rules"]["boundary_faces"]
    with pytest.raises(ValueError, match="Missing or invalid 'boundary_faces'"):
        validate_config(cfg)

def test_ghost_rules_wrong_types_raise():
    cfg = valid_config()
    cfg["ghost_rules"]["face_types"] = "not a dict"
    with pytest.raises(ValueError, match="Missing or invalid 'face_types'"):
        validate_config(cfg)

@pytest.mark.parametrize("bad_type", [None, "string", 42])
def test_boundary_conditions_must_be_list(bad_type):
    cfg = valid_config()
    cfg["boundary_conditions"] = bad_type
    with pytest.raises(ValueError, match="Missing or invalid 'boundary_conditions'"):
        validate_config(cfg)

def test_each_boundary_condition_must_be_dict():
    cfg = valid_config()
    cfg["boundary_conditions"][0] = "not a dict"
    with pytest.raises(ValueError, match="boundary_conditions\

\[0\\]

 must be a dictionary."):
        validate_config(cfg)

def test_missing_apply_to_raises():
    cfg = valid_config()
    del cfg["boundary_conditions"][0]["apply_to"]
    with pytest.raises(ValueError, match="missing or invalid 'apply_to'"):
        validate_config(cfg)

def test_missing_type_raises():
    cfg = valid_config()
    del cfg["boundary_conditions"][0]["type"]
    with pytest.raises(ValueError, match="missing or invalid 'type'"):
        validate_config(cfg)

def test_optional_velocity_wrong_type_raises():
    cfg = valid_config()
    cfg["boundary_conditions"][0]["velocity"] = "not a list"
    with pytest.raises(ValueError, match="'velocity' must be a list"):
        validate_config(cfg)

def test_optional_pressure_wrong_type_raises():
    cfg = valid_config()
    cfg["boundary_conditions"][0]["pressure"] = "not a number"
    with pytest.raises(ValueError, match="'pressure' must be numeric"):
        validate_config(cfg)

def test_optional_no_slip_wrong_type_raises():
    cfg = valid_config()
    cfg["boundary_conditions"][0]["no_slip"] = "not a bool"
    with pytest.raises(ValueError, match="'no_slip' must be boolean"):
        validate_config(cfg)



