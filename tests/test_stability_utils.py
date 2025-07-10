# tests/test_stability_utils.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

SCENARIOS = ["velocity_burst", "cfl_spike", "stable_flow"]
STEP_INDICES = ["step_0000", "step_0001", "step_0002"]

def load_snapshot(scenario_name, step_index):
    filename = f"{scenario_name}_{step_index}.json"
    path = os.path.join(SNAPSHOT_ROOT, filename)
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

        # Required volatility fields
        required_fields = ["max_divergence", "volatility_slope", "volatility_delta", "overflow_detected"]
        for key in required_fields:
            assert key in snap, f"❌ Missing '{key}' in {scenario_name}_{step_index}.json"

        # Type and value checks
        assert snap["max_divergence"] >= 0, f"❌ Invalid divergence in {scenario_name}_{step_index}"
        assert isinstance(snap["volatility_slope"], str), f"❌ volatility_slope must be string in {scenario_name}_{step_index}"
        assert isinstance(snap["volatility_delta"], (int, float)), f"❌ volatility_delta must be numeric in {scenario_name}_{step_index}"
        assert isinstance(snap["overflow_detected"], bool), f"❌ overflow_detected must be boolean in {scenario_name}_{step_index}"

        divergence_slope.append(snap["volatility_slope"])
        volatility_delta.append(snap["volatility_delta"])
        overflow_flags.append(snap["overflow_detected"])

    # Scenario-specific behavioral assertions
    if scenario_name == "velocity_burst":
        assert any(overflow_flags), "⚠️ Overflow flag should be set in velocity_burst"
        assert any(slope == "increasing" for slope in divergence_slope), "⚠️ Divergence slope should rise in velocity_burst"

    if scenario_name == "cfl_spike":
        assert volatility_delta[-1] > volatility_delta[0], "⚠️ Volatility delta should escalate in cfl_spike"

    if scenario_name == "stable_flow":
        assert not any(overflow_flags), "⚠️ No overflow expected in stable_flow"
        assert all(slope == "flat" for slope in divergence_slope), "⚠️ Divergence slope should stay flat in stable_flow"



