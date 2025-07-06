import os
import json
import numpy as np
import re
import pytest

FULL_OUTPUT_DIR = os.environ.get("FULL_OUTPUT_DIR", "data/output_full")
RESTARTED_OUTPUT_DIR = os.environ.get("RESTARTED_OUTPUT_DIR", "data/output_restarted")
REL_TOL = 1e-6
ABS_TOL = 1e-9


def find_common_step_file(full_dir, restart_dir):
    def get_step_nums(directory):
        step_files = os.listdir(directory)
        steps = []
        for name in step_files:
            m = re.match(r"step_(\d+)\.json$", name)
            if m:
                steps.append(int(m.group(1)))
        return sorted(steps)

    full_steps = get_step_nums(full_dir)
    restarted_steps = get_step_nums(restart_dir)
    common = sorted(set(full_steps).intersection(restarted_steps))
    return f"step_{common[-1]:04d}.json" if common else None


def load_json_field(path):
    with open(path, "r") as f:
        data = json.load(f)
    velocity = np.array(data["velocity"])
    pressure = np.array(data["pressure"])
    return velocity, pressure


def test_restart_consistency_on_latest_common_step():
    full_fields = os.path.join(FULL_OUTPUT_DIR, "fields")
    restarted_fields = os.path.join(RESTARTED_OUTPUT_DIR, "fields")

    assert os.path.isdir(full_fields), f"Missing full fields dir: {full_fields}"
    assert os.path.isdir(restarted_fields), f"Missing restarted fields dir: {restarted_fields}"

    common_step = find_common_step_file(full_fields, restarted_fields)
    assert common_step, "No common step_XXXX.json file found in both outputs"

    full_path = os.path.join(full_fields, common_step)
    restart_path = os.path.join(restarted_fields, common_step)

    v_full, p_full = load_json_field(full_path)
    v_restart, p_restart = load_json_field(restart_path)

    assert v_full.shape == v_restart.shape, f"Velocity shape mismatch at {common_step}"
    assert p_full.shape == p_restart.shape, f"Pressure shape mismatch at {common_step}"

    assert np.allclose(v_full, v_restart, rtol=REL_TOL, atol=ABS_TOL), \
        f"Velocity mismatch at {common_step}"

    assert np.allclose(p_full, p_restart, rtol=REL_TOL, atol=ABS_TOL), \
        f"Pressure mismatch at {common_step}"



