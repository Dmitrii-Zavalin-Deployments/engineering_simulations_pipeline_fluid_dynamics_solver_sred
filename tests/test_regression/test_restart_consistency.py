import os
import json
import numpy as np
import pytest

# Replace these if your CI/environment names the folders differently
CONTINUOUS_DIR = os.environ.get("FULL_OUTPUT_DIR", "data/output_full")
RESTARTED_DIR = os.environ.get("RESTARTED_OUTPUT_DIR", "data/output_restarted")
TARGET_STEP = "step_0050.json"
REL_TOL = 1e-6
ABS_TOL = 1e-9


def load_json_field(path):
    with open(path, "r") as f:
        data = json.load(f)
    velocity = np.array(data["velocity"])
    pressure = np.array(data["pressure"])
    return velocity, pressure


@pytest.mark.skipif(
    not (os.path.exists(os.path.join(CONTINUOUS_DIR, TARGET_STEP)) and
         os.path.exists(os.path.join(RESTARTED_DIR, TARGET_STEP))),
    reason="Missing target output files for restart consistency test"
)
def test_restart_output_matches_continuous_run():
    full_path = os.path.join(CONTINUOUS_DIR, TARGET_STEP)
    restart_path = os.path.join(RESTARTED_DIR, TARGET_STEP)

    v_full, p_full = load_json_field(full_path)
    v_restart, p_restart = load_json_field(restart_path)

    assert v_full.shape == v_restart.shape, "Velocity shapes differ"
    assert p_full.shape == p_restart.shape, "Pressure shapes differ"

    assert np.allclose(v_full, v_restart, rtol=REL_TOL, atol=ABS_TOL), \
        "Velocity mismatch between full and restarted run"

    assert np.allclose(p_full, p_restart, rtol=REL_TOL, atol=ABS_TOL), \
        "Pressure mismatch between full and restarted run"



