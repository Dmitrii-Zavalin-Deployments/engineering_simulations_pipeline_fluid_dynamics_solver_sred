# tests/test_pipeline_behavior.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

def discover_snapshot_files():
    """Finds all snapshot files matching the pattern *_step_0000.json"""
    if not os.path.isdir(SNAPSHOT_ROOT):
        return []
    return [
        file for file in os.listdir(SNAPSHOT_ROOT)
        if file.endswith("_step_0000.json")
    ]

SNAPSHOT_FILES = discover_snapshot_files()
skip_reason = "❌ Snapshot files not found — run simulation before testing."

@pytest.mark.parametrize("snapshot_file", SNAPSHOT_FILES or ["_placeholder"])
def test_pipeline_snapshot_behavior(snapshot_file):
    if snapshot_file == "_placeholder":
        pytest.skip(skip_reason)

    snapshot_path = os.path.join(SNAPSHOT_ROOT, snapshot_file)
    assert os.path.isfile(snapshot_path), f"❌ File missing: {snapshot_path}"

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    if "max_divergence" not in snapshot:
        pytest.skip("⚠️ Skipping due to missing snapshot field — regenerate required")

    required_fields = [
        "step_index", "grid", "max_velocity", "max_divergence",
        "global_cfl", "overflow_detected", "damping_enabled", "projection_passes"
    ]
    for field in required_fields:
        assert field in snapshot, f"❌ Missing '{field}' in {snapshot_file}"

    assert isinstance(snapshot["step_index"], int), f"❌ 'step_index' must be int in {snapshot_file}"
    assert isinstance(snapshot["grid"], list), f"❌ 'grid' must be list in {snapshot_file}"
    assert isinstance(snapshot["max_velocity"], (int, float)), f"❌ 'max_velocity' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["max_divergence"], (int, float)), f"❌ 'max_divergence' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["global_cfl"], (int, float)), f"❌ 'global_cfl' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["overflow_detected"], bool), f"❌ 'overflow_detected' must be boolean in {snapshot_file}"
    assert isinstance(snapshot["damping_enabled"], bool), f"❌ 'damping_enabled' must be boolean in {snapshot_file}"
    assert isinstance(snapshot["projection_passes"], int), f"❌ 'projection_passes' must be int in {snapshot_file}"

    assert snapshot["max_velocity"] >= 0, f"❌ Negative 'max_velocity' in {snapshot_file}"
    assert snapshot["max_divergence"] >= 0, f"❌ Negative 'max_divergence' in {snapshot_file}"
    assert snapshot["global_cfl"] >= 0, f"❌ Negative 'global_cfl' in {snapshot_file}"
    assert snapshot["projection_passes"] >= 1, f"❌ Invalid 'projection_passes' in {snapshot_file}"

    # 🔍 Damping-aware logging
    if snapshot.get("damping_enabled") and snapshot["max_velocity"] < 1e-4:
        print(f"🔕 Snapshot suppressed velocity under damping: {snapshot['max_velocity']} in {snapshot_file}")

    scenario = snapshot_file.split("_step_")[0]
    if "cfl_spike" in scenario:
        assert snapshot["damping_enabled"], f"⚠️ Expected damping for CFL spike in {snapshot_file}"
    if "velocity_burst" in scenario:
        assert snapshot["overflow_detected"], f"⚠️ Overflow expected in velocity burst in {snapshot_file}"
        assert snapshot["damping_enabled"], f"⚠️ Damping expected in velocity burst in {snapshot_file}"
    if "projection_overload" in scenario:
        assert snapshot["projection_passes"] > 1, f"⚠️ Projection escalation expected in {snapshot_file}"
    if "stable_flow" in scenario:
        assert not snapshot["overflow_detected"], f"⚠️ No overflow expected in stable flow: {snapshot_file}"
        assert not snapshot["damping_enabled"], f"⚠️ No damping expected in stable flow: {snapshot_file}"



