# tests/test_threshold_validation.py

import os
import json
import pytest
from jsonschema import validate

THRESHOLD_PATH = "tests/test_thresholds.json"
SCHEMA_PATH = "tests/schema/thresholds.schema.json"

def load_json(path):
    assert os.path.isfile(path), f"Missing file: {path}"
    with open(path) as f:
        return json.load(f)

def test_thresholds_match_schema():
    config = load_json(THRESHOLD_PATH)
    schema = load_json(SCHEMA_PATH)
    validate(instance=config, schema=schema)

def test_no_fallbacks_present():
    config = load_json(THRESHOLD_PATH)
    for section, entries in config.items():
        for key, value in entries.items():
            assert value != -1.0, f"Fallback detected at {section}.{key} = -1.0"

def test_required_reflex_keys_present():
    damping = load_json(THRESHOLD_PATH).get("damping_tests", {})
    required = [
        "damping_enabled",
        "damping_factor",
        "abort_cfl_threshold",
        "abort_divergence_threshold",
        "abort_velocity_threshold",
        "divergence_spike_factor",
        "projection_passes_max",
        "max_consecutive_failures"
    ]
    for key in required:
        assert key in damping, f"Missing reflex key: {key}"

def test_numeric_thresholds_are_positive():
    config = load_json(THRESHOLD_PATH)
    for section, entries in config.items():
        for key, value in entries.items():
            if isinstance(value, (int, float)):
                assert value >= 0, f"Negative or invalid value at {section}.{key}: {value}"



