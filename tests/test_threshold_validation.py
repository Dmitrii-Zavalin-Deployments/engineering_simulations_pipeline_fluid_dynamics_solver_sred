# tests/test_threshold_validation.py

import os
import json
import pytest

SCENARIO_DIR = "tests/inputs"

# Required reflex-related keys that must be present in every scenario input
REQUIRED_KEYS = [
    "abort_cfl_threshold",
    "abort_divergence_threshold",
    "abort_velocity_threshold",
    "damping_enabled",
    "damping_factor",
    "divergence_spike_factor",
    "projection_passes",
    "max_consecutive_failures"
]

def load_json(path):
    assert os.path.isfile(path), f"❌ File missing: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario_file", os.listdir(SCENARIO_DIR))
def test_reflex_threshold_structure_and_values(scenario_file):
    scenario_path = os.path.join(SCENARIO_DIR, scenario_file)
    config = load_json(scenario_path)

    for key in REQUIRED_KEYS:
        assert key in config, f"❌ Missing required key '{key}' in {scenario_file}"
        value = config[key]

        # Detect fallback value
        assert value != -1.0, f"⚠️ Fallback (-1.0) detected for '{key}' in {scenario_file}"

        # Value type and range checks
        if key == "damping_enabled":
            assert isinstance(value, bool), f"❌ '{key}' must be boolean in {scenario_file}"
        elif isinstance(value, (int, float)):
            assert value >= 0, f"❌ Invalid negative value for '{key}' in {scenario_file}"
        else:
            assert False, f"❌ Unexpected type for '{key}' in {scenario_file}: {type(value)}"



