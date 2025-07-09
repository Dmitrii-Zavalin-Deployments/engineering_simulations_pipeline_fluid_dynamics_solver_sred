# tests/test_scheduler_reflex.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-output-run/navier_stokes_output"
SCENARIOS = {
    "cfl_spike.json": {
        "expect_reflex": True,
        "expect_damping": True
    },
    "projection_overload.json": {
        "expect_reflex": True,
        "min_projection_passes": 2
    },
    "velocity_burst.json": {
        "expect_reflex": True,
        "expect_damping": True,
        "expect_overflow": True
    },
    "stable_flow.json": {
        "expect_reflex": False,
        "expect_damping": False,
        "expect_overflow": False
    }
}

def load_snapshot(scenario_file, step_file="step_0000.json"):
    path = os.path.join(SNAPSHOT_ROOT, scenario_file.replace(".json", ""), step_file)
    assert os.path.isfile(path), f"❌ Snapshot not found: {path}"
    with open(path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario", SCENARIOS.keys())
def test_reflex_response_and_flags(scenario):
    snap = load_snapshot(scenario)

    expected = SCENARIOS[scenario]

    # Confirm expected keys exist in snapshot
    for key in ["reflex_triggered", "projection_passes", "overflow_flag", "damping_applied"]:
        assert key in snap, f"❌ Missing key '{key}' in snapshot from {scenario}"

    # Reflex trigger check
    assert snap["reflex_triggered"] == expected["expect_reflex"], (
        f"⚠️ Reflex trigger mismatch in {scenario}: expected {expected['expect_reflex']}, got {snap['reflex_triggered']}"
    )

    # Damping logic check (if applicable)
    if "expect_damping" in expected:
        assert snap["damping_applied"] == expected["expect_damping"], (
            f"⚠️ Damping mismatch in {scenario}: expected {expected['expect_damping']}, got {snap['damping_applied']}"
        )

    # Projection escalation (if applicable)
    if "min_projection_passes" in expected:
        actual_passes = snap["projection_passes"]
        assert actual_passes >= expected["min_projection_passes"], (
            f"⚠️ Projection escalation too low in {scenario}: got {actual_passes}, expected ≥ {expected['min_projection_passes']}"
        )

    # Overflow trigger check (if applicable)
    if "expect_overflow" in expected:
        assert snap["overflow_flag"] == expected["expect_overflow"], (
            f"⚠️ Overflow mismatch in {scenario}: expected {expected['expect_overflow']}, got {snap['overflow_flag']}"
        )



