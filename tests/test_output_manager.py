# tests/test_output_manager.py

import os
import json
import pytest

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

EXPECTED_KEYS = [
    "step",
    "grid",
    "max_velocity",
    "max_divergence",
    "global_cfl",
    "overflow_detected",
    "damping_enabled",
    "projection_passes"
]

def discover_snapshots():
    """Returns a list of snapshot files matching *_step_0000.json"""
    return [
        filename for filename in os.listdir(SNAPSHOT_ROOT)
        if filename.endswith("_step_0000.json")
    ]

@pytest.mark.parametrize("snapshot_file", discover_snapshots())
def test_all_snapshots_have_expected_schema_and_values(snapshot_file):
    snapshot_path = os.path.join(SNAPSHOT_ROOT, snapshot_file)
    assert os.path.isfile(snapshot_path), f"❌ Snapshot missing: {snapshot_path}"

    with open(snapshot_path) as f:
        snapshot = json.load(f)

    # Required schema keys
    for key in EXPECTED_KEYS:
        assert key in snapshot, f"❌ Missing key '{key}' in {snapshot_file}"

    # Grid structure
    assert isinstance(snapshot["grid"], list), f"❌ 'grid' should be a list in {snapshot_file}"
    for cell in snapshot["grid"]:
        assert isinstance(cell, list) and len(cell) == 5, f"❌ Grid cell malformed in {snapshot_file}"
        x, y, z, velocity, pressure = cell
        assert all(isinstance(coord, int) for coord in [x, y, z]), f"❌ Grid coordinates must be integers in {snapshot_file}"
        assert isinstance(velocity, list) and len(velocity) == 3, f"❌ Velocity must be 3D vector in {snapshot_file}"
        assert all(isinstance(v, (int, float)) for v in velocity), f"❌ Velocity values must be numeric in {snapshot_file}"
        assert isinstance(pressure, (int, float)), f"❌ Pressure must be numeric in {snapshot_file}"

    # Metadata type validation
    assert isinstance(snapshot["step"], int), f"❌ 'step' must be int in {snapshot_file}"
    assert isinstance(snapshot["max_velocity"], (int, float)), f"❌ 'max_velocity' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["max_divergence"], (int, float)), f"❌ 'max_divergence' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["global_cfl"], (int, float)), f"❌ 'global_cfl' must be numeric in {snapshot_file}"
    assert isinstance(snapshot["overflow_detected"], bool), f"❌ 'overflow_detected' must be boolean in {snapshot_file}"
    assert isinstance(snapshot["damping_enabled"], bool), f"❌ 'damping_enabled' must be boolean in {snapshot_file}"
    assert isinstance(snapshot["projection_passes"], int), f"❌ 'projection_passes' must be int in {snapshot_file}"

    # Value bounds
    assert snapshot["max_velocity"] >= 0, f"❌ Invalid max_velocity in {snapshot_file}"
    assert snapshot["max_divergence"] >= 0, f"❌ Invalid max_divergence in {snapshot_file}"
    assert snapshot["global_cfl"] >= 0, f"❌ global_cfl must be non-negative in {snapshot_file}"
    assert snapshot["projection_passes"] >= 0, f"❌ projection_passes must be non-negative in {snapshot_file}"



