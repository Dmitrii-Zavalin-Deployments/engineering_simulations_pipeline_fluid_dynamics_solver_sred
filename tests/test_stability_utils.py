# tests/test_stability_utils.py

import os
import json
import pytest

SNAPSHOT_DIR = "data/testing-input-output/navier_stokes_output"
SCENARIOS = [
    "velocity_burst",
    "cfl_spike",
    "stable_flow"
]

STEP_INDICES = ["step_0000", "step_0001", "step_0002"]

def load_snapshot(scenario_name, step_index):
    filename = f"{scenario_name}_{step_index}.json"
    path = os.path.join(SNAPSHOT_DIR, filename)
    assert os.path.isfile(path), f"❌ Snapshot missing: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario_name", SCENARIOS)
def test_volatility_metrics_across_timesteps(scenario_name):
    divergence_slope = []
    volatility_delta = []
    overflow_flags = []

    for step_index in STEP_INDICES:
        snap = load_snapshot(scenario_name, step_index)

        # Required fields
        for key in ["max_divergence", "volatility_slope", "volatility_delta", "overflow_detected"]:
            assert key in snap, f"❌ Missing '{key}' in {scenario_name}_{step_index}.json"

        divergence = snap["max_divergence"]
        slope = snap["volatility_slope"]
        delta = snap["volatility_delta"]
        overflow = snap["overflow_detected"]

        # Sanity checks
        assert divergence >= 0, f"❌ Invalid divergence value in {scenario_name}_{step_index}"
        assert isinstance(slope, str), f"❌ volatility_slope must be a string in {scenario_name}_{step_index}"
        assert isinstance(delta, (int, float)), f"❌ volatility_delta must be numeric in {scenario_name}_{step_index}"
        assert isinstance(overflow, bool), f"❌ overflow_detected must be boolean in {scenario_name}_{step_index}"

        divergence_slope.append(slope)
        volatility_delta.append(delta)
        overflow_flags.append(overflow)

    # Scenario-specific expectations
    if scenario_name == "velocity_burst":
        assert any(overflow_flags), "⚠️ Expected overflow flag in velocity_burst"
        assert any(s == "increasing" for s in divergence_slope), "⚠️ Divergence slope should be increasing in velocity burst"

    if scenario_name == "cfl_spike":
        assert volatility_delta[-1] > volatility_delta[0], "⚠️ Volatility delta should grow under CFL spike"

    if scenario_name == "stable_flow":
        assert not any(overflow_flags), "⚠️ No overflow expected in stable_flow"
        assert all(s == "flat" for s in divergence_slope), "⚠️ Divergence slope should remain flat in stable_flow"



