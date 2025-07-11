# tests/test_output_manager.py

import os
import json
import pytest
from datetime import datetime

SNAPSHOT_ROOT = "data/testing-input-output/navier_stokes_output"

EXPECTED_KEYS = [
    "step_index",
    "grid",
    "max_velocity",
    "max_divergence",
    "global_cfl",
    "overflow_detected",
    "damping_enabled",
    "projection_passes"
]

def discover_snapshots():
    """Returns snapshot filenames matching *_step_*.json"""
    if not os.path.isdir(SNAPSHOT_ROOT):
        return []
    return [
        filename for filename in os.listdir(SNAPSHOT_ROOT)
        if filename.endswith(".json") and "_step_" in filename
    ]

@pytest.mark.parametrize("snapshot_file", discover_snapshots())
def test_snapshot_structure_and_values(snapshot_file):
    path = os.path.join(SNAPSHOT_ROOT, snapshot_file)
    assert os.path.isfile(path), f"❌ File missing: {path}"

    with open(path) as f:
        snap = json.load(f)

    # Required keys
    for key in EXPECTED_KEYS:
        assert key in snap, f"❌ Missing key '{key}' in {snapshot_file}"

    # Grid checks
    assert isinstance(snap["grid"], list), f"❌ Grid must be a list in {snapshot_file}"
    for cell in snap["grid"]:
        assert isinstance(cell, dict), f"❌ Grid cell must be a dict in {snapshot_file}"
        assert all(k in cell for k in ["x", "y", "z", "velocity", "pressure"]), f"❌ Missing cell keys in {snapshot_file}"
        assert isinstance(cell["x"], (int, float))
        assert isinstance(cell["y"], (int, float))
        assert isinstance(cell["z"], (int, float))
        assert isinstance(cell["pressure"], (int, float))
        assert isinstance(cell["velocity"], list) and len(cell["velocity"]) == 3
        assert all(isinstance(v, (int, float)) for v in cell["velocity"])

    # Metric types
    assert isinstance(snap["step_index"], int), f"❌ step_index must be int"
    assert isinstance(snap["max_velocity"], (int, float)), f"❌ max_velocity must be numeric"
    assert isinstance(snap["max_divergence"], (int, float)), f"❌ max_divergence must be numeric"
    assert isinstance(snap["global_cfl"], (int, float)), f"❌ global_cfl must be numeric"
    assert isinstance(snap["overflow_detected"], bool), f"❌ overflow_detected must be bool"
    assert isinstance(snap["damping_enabled"], bool), f"❌ damping_enabled must be bool"
    assert isinstance(snap["projection_passes"], int), f"❌ projection_passes must be int"

    # Value bounds
    assert snap["max_velocity"] >= 0
    assert snap["max_divergence"] >= 0
    assert snap["global_cfl"] >= 0
    assert snap["projection_passes"] >= 1



