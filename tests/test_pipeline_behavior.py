# tests/test_pipeline_behavior.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-output-run/navier_stokes_output"

# Each scenario maps to a directory, within which we expect at least step_0000.json
SCENARIOS = [
    "stable_flow.json",
    "cfl_spike.json",
    "projection_overload.json",
    "velocity_burst.json",
    "damped_cavity.json"
]

def load_snapshot(scenario_name, step="step_0000.json"):
    snapshot_path = os.path.join(SNAPSHOT_ROOT, scenario_name.replace(".json", ""), step)
    assert os.path.isfile(snapshot_path), f"❌ Snapshot missing: {snapshot_path}"
    with open(snapshot_path) as f:
        return json.load(f)

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_pipeline_snapshot_behavior(scenario):
    snapshot = load_snapshot(scenario)

    # Core behavioral assertions
    assert "divergence_max" in snapshot, f"❌ Missing divergence metric in {scenario}"
    assert "velocity_max" in snapshot, f"❌ Missing velocity metric in {scenario}"
    assert "overflow_flag" in snapshot, f"❌ Missing overflow flag in {scenario}"
    assert "projection_passes" in snapshot, f"❌ Missing projection depth in {scenario}"
    assert "reflex_triggered" in snapshot, f"❌ Missing reflex trigger status in {scenario}"

    # Sanity checks
    assert snapshot["divergence_max"] >= 0, f"❌ Invalid divergence_max value in {scenario}"
    assert snapshot["velocity_max"] >= 0, f"❌ Invalid velocity_max value in {scenario}"
    assert isinstance(snapshot["overflow_flag"], bool), f"❌ overflow_flag should be boolean in {scenario}"
    assert isinstance(snapshot["reflex_triggered"], bool), f"❌ reflex_triggered should be boolean in {scenario}"
    assert snapshot["projection_passes"] >= 0, f"❌ projection_passes should be ≥ 0 in {scenario}"

    # Reflex and escalation logic expectations (optional and extendable)
    if scenario == "cfl_spike.json":
        assert snapshot["reflex_triggered"], "⚠️ CFL spike should trigger reflex"
        assert snapshot.get("damping_applied", False), "⚠️ Damping expected to be applied"

    if scenario == "velocity_burst.json":
        assert snapshot["overflow_flag"], "⚠️ Velocity burst should trigger overflow"
        assert snapshot["reflex_triggered"], "⚠️ Reflex expected due to velocity spike"

    if scenario == "projection_overload.json":
        assert snapshot["projection_passes"] > 1, "⚠️ Projection escalation expected"

    if scenario == "stable_flow.json":
        assert not snapshot["reflex_triggered"], "⚠️ Reflex should not trigger in stable flow"
        assert not snapshot["overflow_flag"], "⚠️ No overflow expected in stable flow"



