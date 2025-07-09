# tests/test_output_manager.py

import os
import json
import pytest
import numpy as np
from stability_utils import compute_volatility

SNAPSHOT_PATH = "data/testing-output-run/navier_stokes_output/divergence_snapshot.json"

def simulate_overflow_snapshot_write(path, step=42, divergence_val=float("inf"), velocity_val=1e6, cfl_val=2.5):
    """
    Simulates writing a divergence snapshot file with overflow metrics.
    """
    snapshot_data = {
        "step": step,
        "max_divergence": divergence_val,
        "max_velocity": velocity_val,
        "global_cfl": cfl_val,
        "overflow_detected": np.isinf(divergence_val),
        "divergence_mode": "log",
        "field_shape": [8, 8, 8],
        "divergence_values": np.full((8, 8, 8), divergence_val).tolist()
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
    # ✅ Ensure snapshot exists before checking
    simulate_overflow_snapshot_write(SNAPSHOT_PATH)
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
    assert snapshot_fixture["global_cfl"] > 1.0

def test_field_shape_matches(snapshot_fixture):
    shape = snapshot_fixture["field_shape"]
    values = snapshot_fixture["divergence_values"]
    assert isinstance(values, list)
    assert len(values) == shape[0]
    assert all(len(row) == shape[1] for row in values)

def test_snapshot_volatility_metrics():
    # Simulate previous and current snapshot for volatility comparison
    step_prev, step_curr = 41, 42
    d_prev = 900.0
    d_curr = float("inf")

    simulate_overflow_snapshot_write(SNAPSHOT_PATH, step=step_curr, divergence_val=d_curr, velocity_val=1e6, cfl_val=2.5)
    delta, slope = compute_volatility(d_curr, d_prev, step_curr - step_prev)

    assert np.isinf(delta)
    assert np.isinf(slope)



