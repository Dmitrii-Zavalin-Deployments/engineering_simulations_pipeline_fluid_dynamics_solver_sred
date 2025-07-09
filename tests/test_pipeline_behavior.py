# tests/test_pipeline_behavior.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

def discover_snapshots():
    """Returns a list of snapshot files ending with _step_0000.json"""
    return [
        filename for filename in os.listdir(SNAPSHOT_ROOT)
        if filename.endswith("_step_0000.json")
    ]

@pytest.mark.parametrize("snapshot_file", discover_snapshots())
def test_pipeline_snapshot_behavior(snapshot_file):
    snapshot_path = os.path.join(SNAPSHOT_ROOT, snapshot_file)
    with open(snapshot_path) as f:
        snapshot = json.load(f)

    # Basic structural fields
    assert "step" in snapshot, f"❌ Missing 'step' in {snapshot_file}"
    assert "grid" in snapshot, f"❌ Missing 'grid' in {snapshot_file}"
    assert isinstance(snapshot["grid"], list), f"❌ 'grid' must be a list in {snapshot_file}"
    assert "max_velocity" in snapshot, f"❌ Missing 'max_velocity' in {snapshot_file}"
    assert "max_divergence" in snapshot, f"❌ Missing 'max_divergence' in {snapshot_file}"
    assert "global_cfl" in snapshot, f"❌ Missing 'global_cfl' in {snapshot_file}"
    assert "overflow_detected" in snapshot, f"❌ Missing 'overflow_detected' in {snapshot_file}"
    assert "damping_enabled" in snapshot, f"❌ Missing 'damping_enabled' in {snapshot_file}"
    assert "projection_passes" in snapshot, f"❌ Missing 'projection_passes' in {snapshot_file}"

    # Type checks
    assert isinstance(snapshot["step"], int), f"❌ 'step' must be int in {snapshot_file}"
    assert isinstance(snapshot["max_velocity"], (int, float)), f"❌ 'max_velocity' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["max_divergence"], (int, float)), f"❌ 'max_divergence' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["global_cfl"], (int, float)), f"❌ 'global_cfl' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["overflow_detected"], bool), f"❌ 'overflow_detected' must be boolean in {snapshot_file}"
    assert isinstance(snapshot["damping_enabled"], bool), f"❌ 'damping_enabled' must be boolean in {snapshot_file}"
    assert isinstance(snapshot["projection_passes"], int), f"❌ 'projection_passes' must be int in {snapshot_file}"

    # Value bounds
    assert snapshot["max_velocity"] >= 0, f"❌ Negative 'max_velocity' in {snapshot_file}"
    assert snapshot["max_divergence"] >= 0, f"❌ Negative 'max_divergence' in {snapshot_file}"
    assert snapshot["global_cfl"] >= 0, f"❌ Negative 'global_cfl' in {snapshot_file}"
    assert snapshot["projection_passes"] >= 0, f"❌ Invalid 'projection_passes' in {snapshot_file}"

    # Optional logic extensions
    name = snapshot_file.split("_step_")[0]
    if "cfl_spike" in name:
        assert snapshot["damping_enabled"], f"⚠️ Expected damping for CFL spike in {snapshot_file}"

    if "velocity_burst" in name:
        assert snapshot["overflow_detected"], f"⚠️ Velocity burst should trigger overflow in {snapshot_file}"
        assert snapshot["damping_enabled"], f"⚠️ Reflex expected due to velocity spike in {snapshot_file}"

    if "projection_overload" in name:
        assert snapshot["projection_passes"] > 1, f"⚠️ Projection overload should escalate in {snapshot_file}"

    if "stable_flow" in name:
        assert not snapshot["overflow_detected"], f"⚠️ No overflow expected in stable flow: {snapshot_file}"
        assert not snapshot["damping_enabled"], f"⚠️ No damping expected in stable flow: {snapshot_file}"



