import json
import pytest
import numpy as np
from stability_utils import compute_volatility

def load_snapshot(path):
    with open(path) as f:
        return json.load(f)

def test_snapshot_volatility_across_timesteps():
    s0 = load_snapshot("tests/inputs/stable_flow.json")
    s1 = load_snapshot("tests/inputs/cfl_spike.json")
    s2 = load_snapshot("tests/inputs/velocity_burst.json")

    delta_01, slope_01 = compute_volatility(s1["max_divergence"], s0["max_divergence"], s1["step"] - s0["step"])
    delta_12, slope_12 = compute_volatility(s2["max_divergence"], s1["max_divergence"], s2["step"] - s1["step"])

    assert slope_01 > 0
    assert slope_12 > slope_01
    assert np.isfinite(slope_01)
    assert np.isfinite(delta_12)



