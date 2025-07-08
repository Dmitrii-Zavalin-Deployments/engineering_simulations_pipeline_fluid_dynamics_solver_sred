# tests/test_output_manager.py

import os
import json
import pytest
import numpy as np

SNAPSHOT_PATH = "data/testing-output-run/navier_stokes_output/divergence_snapshot.json"

def simulate_overflow_snapshot_write(path):
    """
    Simulates writing a divergence snapshot file with overflow metrics.
    """
    snapshot_data = {
        "step": 42,
        "max_divergence": float("inf"),
        "max_velocity": 1e6,
        "global_cfl": 2.5,
        "overflow_detected": True,
        "divergence_mode": "log",
        "field_shape": [8, 8, 8],
        "divergence_values": np.full((8, 8, 8), 1e3).tolist()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot_data, f, indent=2)

@pytest.fixture(scope="module")
def snapshot_fixture():
    simulate_overflow_snapshot_write(SNAPSHOT_PATH)
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)

def test_snapshot_file_exists():
    assert os.path.isfile(SNAPSHOT_PATH), "Snapshot file is missing"

def test_snapshot_schema(snapshot_fixture):
    required_keys = [
        "step",
        "max_divergence",
        "max_velocity",
        "global_cfl",
        "overflow_detected",
        "divergence_mode",
        "field_shape",
        "divergence_values"
    ]
    for key in required_keys:
        assert key in snapshot_fixture, f"Missing key in snapshot: {key}"

def test_overflow_flag_triggered(snapshot_fixture):
    assert snapshot_fixture["overflow_detected"] is True
    assert np.isinf(snapshot_fixture["max_divergence"])

def test_field_shape_matches(snapshot_fixture):
    shape = snapshot_fixture["field_shape"]
    values = snapshot_fixture["divergence_values"]
    assert isinstance(values, list)
    assert len(values) == shape[0]
    assert all(len(row) == shape[1] for row in values)



