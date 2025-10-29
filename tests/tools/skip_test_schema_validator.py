import pytest
import json
from pathlib import Path
from src.tools import schema_validator
from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions

@pytest.fixture
def valid_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "boundary_conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "apply_to": {"type": "string"},
                        "type": {"type": "string"}
                    },
                    "required": ["apply_to"]
                }
            },
            "ghost_rules": {
                "type": "object",
                "properties": {
                    "boundary_faces": {"type": "array"},
                    "default_type": {"type": "string"},
                    "face_types": {"type": "object"}
                },
                "required": ["boundary_faces", "default_type", "face_types"]
            }
        },
        "required": ["boundary_conditions"]
    }

@pytest.fixture
def valid_instance():
    return {
        "boundary_conditions": [
            {"apply_to": "inlet", "type": "Dirichlet"},
            {"apply_to": "wall", "type": "no-slip"}
        ],
        "ghost_rules": {
            "boundary_faces": ["x+", "x-"],
            "default_type": "ghost",
            "face_types": {"x+": "ghost", "x-": "ghost"}
        }
    }

def test_valid_schema_passes(valid_schema, valid_instance):
    # Should not raise any exceptions
    schema_validator.validate_schema(valid_instance, valid_schema)

def test_missing_apply_to_key_fails(valid_schema):
    instance = {
        "boundary_conditions": [{"type": "Dirichlet"}]
    }
    with pytest.raises(SystemExit):
        schema_validator.validate_schema(instance, valid_schema)

def test_boundary_conditions_not_list_fails(valid_schema):
    instance = {
        "boundary_conditions": {"apply_to": "inlet"}
    }
    with pytest.raises(SystemExit):
        schema_validator.validate_schema(instance, valid_schema)

def test_boundary_condition_not_dict_fails(valid_schema):
    instance = {
        "boundary_conditions": ["inlet"]
    }
    with pytest.raises(SystemExit):
        schema_validator.validate_schema(instance, valid_schema)

def test_missing_ghost_rules_key_fails(valid_schema):
    instance = {
        "boundary_conditions": [{"apply_to": "inlet"}],
        "ghost_rules": {
            "boundary_faces": ["x+"],
            "default_type": "ghost"
            # Missing "face_types"
        }
    }
    with pytest.raises(SystemExit):
        schema_validator.validate_schema(instance, valid_schema)

def test_ghost_rules_not_dict_fails(valid_schema):
    instance = {
        "boundary_conditions": [{"apply_to": "inlet"}],
        "ghost_rules": ["x+", "x-"]
    }
    with pytest.raises(SystemExit):
        schema_validator.validate_schema(instance, valid_schema)

def test_load_json_valid(tmp_path):
    test_file = tmp_path / "valid.json"
    test_file.write_text(json.dumps({"key": "value"}))
    result = schema_validator.load_json(str(test_file))
    assert result == {"key": "value"}

def test_load_json_invalid(tmp_path):
    test_file = tmp_path / "invalid.json"
    test_file.write_text("{ invalid json }")
    with pytest.raises(SystemExit):
        schema_validator.load_json(str(test_file))



