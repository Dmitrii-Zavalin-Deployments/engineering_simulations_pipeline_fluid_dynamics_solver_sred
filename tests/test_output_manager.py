# tests/test_output_manager.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-output-run/navier_stokes_output"
EXPECTED_KEYS = [
    "divergence_max",
    "velocity_max",
    "overflow_flag",
    "reflex_triggered",
    "projection_passes",
    "volatility_slope",
    "volatility_delta",
    "damping_applied",
    "step_index",
    "timestamp"
]

SCENARIOS = [
    "stable_flow.json",
    "cfl_spike.json",
    "velocity_burst.json",
    "projection_overload.json",
    "damped_cavity.json"
]

STEP_FILES = ["step_0000.json", "step_0001.json", "step_0002.json"]  # Extendable if needed

def load_snapshot(scenario, step_file):
    path = os.path.join(SNAPSHOT_ROOT, scenario.replace(".json", ""), step_file)
    assert os.path.isfile(path), f"❌ Snapshot missing: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_snapshot_schema_and_behavioral_fields(scenario):
    for step_file in STEP_FILES:
        snapshot = load_snapshot(scenario, step_file)

        # Schema validation
        for key in EXPECTED_KEYS:
            assert key in snapshot, f"❌ Missing key '{key}' in {step_file} of {scenario}"

        # Field type validation
        assert isinstance(snapshot["divergence_max"], (int, float)), "❌ divergence_max should be numeric"
        assert isinstance(snapshot["velocity_max"], (int, float)), "❌ velocity_max should be numeric"
        assert isinstance(snapshot["overflow_flag"], bool), "❌ overflow_flag should be boolean"
        assert isinstance(snapshot["reflex_triggered"], bool), "❌ reflex_triggered should be boolean"
        assert isinstance(snapshot["projection_passes"], int), "❌ projection_passes should be integer"
        assert isinstance(snapshot["volatility_slope"], str), "❌ volatility_slope should be string"
        assert isinstance(snapshot["volatility_delta"], (int, float)), "❌ volatility_delta should be numeric"
        assert isinstance(snapshot["damping_applied"], bool), "❌ damping_applied should be boolean"
        assert isinstance(snapshot["step_index"], int), "❌ step_index should be integer"
        assert isinstance(snapshot["timestamp"], str), "❌ timestamp should be string"

        # Behavioral bounds
        assert snapshot["divergence_max"] >= 0, "❌ Invalid divergence_max value"
        assert snapshot["velocity_max"] >= 0, "❌ Invalid velocity_max value"
        assert snapshot["projection_passes"] >= 0, "❌ projection_passes should be non-negative"
        assert snapshot["volatility_delta"] >= 0, "❌ volatility_delta should be non-negative"



