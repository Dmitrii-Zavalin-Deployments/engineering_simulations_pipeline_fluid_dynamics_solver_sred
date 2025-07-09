# tests/test_stability_pipeline.py

import pytest
import numpy as np
import os
import json
from stability_utils import compute_volatility, check_field_validity

SNAPSHOT_DIR = "data/testing-input-output"
SNAPSHOT_TEMPLATE = {
    "divergence_mode": "log",
    "overflow_detected": False,
    "global_cfl": 0.5,
    "field_shape": [8, 8, 8]
}

def generate_synthetic_snapshot(step, divergence_val, velocity_val, cfl_val, overflow=False):
    """Generates a synthetic snapshot JSON file with step-wise volatility data"""
    snapshot = SNAPSHOT_TEMPLATE.copy()
    snapshot.update({
        "step": step,
        "max_divergence": divergence_val,
        "max_velocity": velocity_val,
        "global_cfl": cfl_val,
        "overflow_detected": overflow,
        "divergence_values": np.full((8, 8, 8), divergence_val).tolist()
    })
    path = os.path.join(SNAPSHOT_DIR, f"step_{step:03d}_snapshot.json")
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return path

@pytest.mark.parametrize("step, prev_val, current_val", [
    (1, 50.0, 55.0),
    (2, 100.0, 90.0),
    (3, 500.0, 500.0)
])
def test_volatility_delta_and_slope(step, prev_val, current_val):
    delta, slope = compute_volatility(current_val, prev_val, step)
    assert isinstance(delta, float)
    assert isinstance(slope, float)
    # Simple check: signs match direction
    if current_val > prev_val:
        assert delta > 0 and slope > 0
    elif current_val < prev_val:
        assert delta < 0 and slope < 0
    else:
        assert delta == 0 and slope == 0

def test_snapshot_field_validity_check():
    path = generate_synthetic_snapshot(step=4, divergence_val=1e3, velocity_val=1e6, cfl_val=0.75)
    with open(path) as f:
        data = json.load(f)
        field = np.array(data["divergence_values"])
        assert field.shape == (8, 8, 8)
        assert check_field_validity(field)

def test_snapshot_volatility_sequence():
    # Generate step 0 and step 1 snapshots
    p0 = generate_synthetic_snapshot(step=0, divergence_val=100.0, velocity_val=900.0, cfl_val=0.7)
    p1 = generate_synthetic_snapshot(step=1, divergence_val=150.0, velocity_val=1000.0, cfl_val=0.9)

    with open(p0) as f0, open(p1) as f1:
        d0 = json.load(f0)
        d1 = json.load(f1)
        delta, slope = compute_volatility(d1["max_divergence"], d0["max_divergence"], step=1)

    assert delta == 50.0
    assert slope == 50.0  # One step difference
    assert isinstance(delta, float)
    assert slope >= 0



