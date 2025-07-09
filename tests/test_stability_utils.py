# tests/test_stability_utils.py

import os
import json
import pytest

SNAPSHOT_DIR = "data/testing-output-run/navier_stokes_output"
SCENARIOS = [
    "velocity_burst.json",
    "cfl_spike.json",
    "stable_flow.json"
]

STEP_FILES = ["step_0000.json", "step_0001.json", "step_0002.json"]

def load_snapshot(scenario, step):
    path = os.path.join(SNAPSHOT_DIR, scenario.replace(".json", ""), step)
    assert os.path.isfile(path), f"❌ Snapshot missing: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_volatility_metrics_across_timesteps(scenario):
    divergence_slope = []
    volatility_delta = []
    overflow_flags = []

    for step in STEP_FILES:
        snap = load_snapshot(scenario, step)

        # Required fields
        for key in ["divergence_max", "volatility_slope", "volatility_delta", "overflow_flag"]:
            assert key in snap, f"❌ Missing '{key}' in {step} of {scenario}"

        divergence = snap["divergence_max"]
        slope = snap["volatility_slope"]
        delta = snap["volatility_delta"]
        overflow = snap["overflow_flag"]

        # Sanity assertions
        assert divergence >= 0, f"❌ Invalid divergence value in {step} of {scenario}"
        assert isinstance(slope, str), f"❌ volatility_slope must be a string in {step} of {scenario}"
        assert isinstance(delta, (int, float)), f"❌ volatility_delta must be numeric in {step} of {scenario}"
        assert isinstance(overflow, bool), f"❌ overflow_flag must be boolean in {step} of {scenario}"

        divergence_slope.append(slope)
        volatility_delta.append(delta)
        overflow_flags.append(overflow)

    # Scenario expectations
    if scenario == "velocity_burst.json":
        assert any(overflow_flags), "⚠️ Expected overflow flag in velocity_burst.json"
        assert any(d == "increasing" for d in divergence_slope), "⚠️ Divergence slope should increase in burst"

    if scenario == "cfl_spike.json":
        assert volatility_delta[-1] > volatility_delta[0], "⚠️ Volatility delta should grow under CFL spike"

    if scenario == "stable_flow.json":
        assert not any(overflow_flags), "⚠️ No overflow expected in stable flow"
        assert all(d == "flat" for d in divergence_slope), "⚠️ Divergence slope should remain flat in stable flow"



