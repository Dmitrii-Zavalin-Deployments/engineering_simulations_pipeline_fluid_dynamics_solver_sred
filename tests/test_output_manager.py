import os
import json
import pytest

SNAPSHOT_PATH = "data/testing-output-run/navier_stokes_output/divergence_snapshot.json"

def test_snapshot_file_exists():
    assert os.path.isfile(SNAPSHOT_PATH), "Snapshot file missing"

def test_snapshot_has_required_keys():
    with open(SNAPSHOT_PATH) as f:
        snap = json.load(f)

    expected_keys = [
        "step", "max_divergence", "max_velocity", "global_cfl",
        "overflow_detected", "damping_enabled", "divergence_mode",
        "divergence_values", "projection_passes"
    ]
    for key in expected_keys:
        assert key in snap, f"Missing key: {key}"

def test_snapshot_structure():
    with open(SNAPSHOT_PATH) as f:
        snap = json.load(f)
        values = snap["divergence_values"]
        assert isinstance(values, list)
        assert len(values) == 8  # Assuming [8,8,8] shape
        for row in values:
            assert isinstance(row, list)
            assert len(row) == 8



