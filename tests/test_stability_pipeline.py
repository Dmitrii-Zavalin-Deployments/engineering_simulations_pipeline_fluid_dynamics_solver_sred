# tests/test_stability_pipeline.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"
STEPS = ["step_0000.json", "step_0001.json", "step_0002.json"]

SCENARIOS = [
    "cfl_spike.json",
    "velocity_burst.json",
    "projection_overload.json"
]

def load_snapshot(scenario, step_file):
    path = os.path.join(SNAPSHOT_ROOT, scenario.replace(".json", ""), step_file)
    assert os.path.isfile(path), f"❌ Snapshot missing: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_stability_over_time(scenario):
    divergence_trend = []
    velocity_trend = []
    projection_passes = []
    reflex_steps = 0
    overflow_steps = 0

    for step_file in STEPS:
        snap = load_snapshot(scenario, step_file)

        # Assert core fields exist
        for key in ["divergence_max", "velocity_max", "projection_passes", "reflex_triggered", "overflow_flag"]:
            assert key in snap, f"❌ '{key}' missing in {step_file} of {scenario}"

        # Track values over time
        divergence_trend.append(snap["divergence_max"])
        velocity_trend.append(snap["velocity_max"])
        projection_passes.append(snap["projection_passes"])

        if snap["reflex_triggered"]:
            reflex_steps += 1
        if snap["overflow_flag"]:
            overflow_steps += 1

    # Basic progression checks
    assert reflex_steps > 0, f"⚠️ Reflex never triggered in scenario '{scenario}'"
    assert all(d >= 0 for d in divergence_trend), f"❌ Negative divergence detected in {scenario}"
    assert all(v >= 0 for v in velocity_trend), f"❌ Negative velocity detected in {scenario}"

    # Scenario-specific expectations
    if scenario == "cfl_spike.json":
        assert reflex_steps >= 2, "⚠️ Reflex should trigger multiple times under sustained CFL instability"

    if scenario == "projection_overload.json":
        assert max(projection_passes) >= 2, "⚠️ Expected projection escalation in projection_overload.json"

    if scenario == "velocity_burst.json":
        assert overflow_steps >= 1, "⚠️ Expected overflow reaction in velocity_burst.json"
        assert any(v > 200 for v in velocity_trend), "⚠️ Velocity burst threshold not reached"



