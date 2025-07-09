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

def load_snapshot(path):
    assert os.path.isfile(path), f"❌ Snapshot missing: {path}"
    with open(path) as f:
        return json.load(f)

def test_all_snapshots_have_expected_schema_and_values():
    for scenario_name in os.listdir(SNAPSHOT_ROOT):
        scenario_path = os.path.join(SNAPSHOT_ROOT, scenario_name)
        if not os.path.isdir(scenario_path):
            continue  # Skip files, only look at folders

        step_file = os.path.join(scenario_path, "step_0000.json")
        snapshot = load_snapshot(step_file)

        # Schema keys presence
        for key in EXPECTED_KEYS:
            assert key in snapshot, f"❌ Missing key '{key}' in {step_file}"

        # Grid structure
        assert isinstance(snapshot["grid"], list), f"❌ 'grid' should be a list in {step_file}"
        for cell in snapshot["grid"]:
            assert isinstance(cell, list) and len(cell) == 5, f"❌ Grid cell malformed in {step_file}"
            x, y, z, velocity, pressure = cell
            assert all(isinstance(coord, int) for coord in [x, y, z]), "❌ Grid coordinates must be integers"
            assert isinstance(velocity, list) and len(velocity) == 3, "❌ Velocity must be a 3D vector"
            assert all(isinstance(v, (int, float)) for v in velocity), "❌ Velocity vector must contain numbers"
            assert isinstance(pressure, (int, float)), "❌ Pressure must be numeric"

        # Metadata types
        assert isinstance(snapshot["step"], int), "❌ 'step' must be an integer"
        assert isinstance(snapshot["max_velocity"], (int, float)), "❌ 'max_velocity' must be numeric"
        assert isinstance(snapshot["max_divergence"], (int, float)), "❌ 'max_divergence' must be numeric"
        assert isinstance(snapshot["global_cfl"], (int, float)), "❌ 'global_cfl' must be numeric"
        assert isinstance(snapshot["overflow_detected"], bool), "❌ 'overflow_detected' must be boolean"
        assert isinstance(snapshot["damping_enabled"], bool), "❌ 'damping_enabled' must be boolean"
        assert isinstance(snapshot["projection_passes"], int), "❌ 'projection_passes' must be integer"

        # Numeric bounds
        assert snapshot["max_velocity"] >= 0, "❌ Invalid max_velocity value"
        assert snapshot["max_divergence"] >= 0, "❌ Invalid max_divergence value"
        assert snapshot["global_cfl"] >= 0, "❌ global_cfl must be non-negative"
        assert snapshot["projection_passes"] >= 0, "❌ projection_passes must be non-negative"



