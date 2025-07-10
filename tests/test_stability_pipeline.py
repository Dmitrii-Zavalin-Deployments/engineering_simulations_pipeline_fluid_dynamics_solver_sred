# tests/test_stability_pipeline.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

STEP_INDICES = ["step_0000", "step_0001", "step_0002"]
SCENARIOS = ["cfl_spike", "velocity_burst", "projection_overload"]

def load_snapshot(scenario_prefix, step_index):
    filename = f"{scenario_prefix}_{step_index}.json"
    path = os.path.join(SNAPSHOT_ROOT, filename)
    assert os.path.isfile(path), f"❌ Snapshot missing: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario_prefix", SCENARIOS)
def test_stability_over_time(scenario_prefix):
    divergence_trend = []
    velocity_trend = []
    projection_passes = []
    damping_flags = []
    overflow_flags = []

    for step_index in STEP_INDICES:
        snap = load_snapshot(scenario_prefix, step_index)

        # Check required fields exist
        required_fields = [
            "max_divergence", "max_velocity", "projection_passes",
            "damping_enabled", "overflow_detected"
        ]
        for key in required_fields:
            assert key in snap, f"❌ Missing '{key}' in {scenario_prefix}_{step_index}.json"

        # Track each metric across steps
        divergence_trend.append(snap["max_divergence"])
        velocity_trend.append(snap["max_velocity"])
        projection_passes.append(snap["projection_passes"])
        damping_flags.append(snap["damping_enabled"])
        overflow_flags.append(snap["overflow_detected"])

    # General checks
    assert all(d >= 0 for d in divergence_trend), f"❌ Negative divergence in {scenario_prefix}"
    assert all(v >= 0 for v in velocity_trend), f"❌ Negative velocity in {scenario_prefix}"
    assert any(projection_passes), f"⚠️ No projection passes logged in {scenario_prefix}"
    assert any(isinstance(flag, bool) for flag in damping_flags), f"⚠️ Damping values must be boolean in {scenario_prefix}"
    assert any(isinstance(flag, bool) for flag in overflow_flags), f"⚠️ Overflow values must be boolean in {scenario_prefix}"

    # Scenario-specific expectations
    if scenario_prefix == "cfl_spike":
        assert damping_flags.count(True) >= 2, "⚠️ Damping should trigger ≥ 2 times under CFL spike"

    if scenario_prefix == "projection_overload":
        assert max(projection_passes) >= 2, "⚠️ Expected projection escalation in projection_overload"

    if scenario_prefix == "velocity_burst":
        assert any(overflow_flags), "⚠️ Expected overflow in velocity_burst"
        assert any(v > 2.0 for v in velocity_trend), "⚠️ Velocity burst threshold not reached in velocity_burst"



