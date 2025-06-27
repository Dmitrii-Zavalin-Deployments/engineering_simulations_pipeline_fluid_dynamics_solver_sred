import os
import json
import numpy as np
import pytest
import re

OUTPUT_DIR = os.environ.get("ACTUAL_OUTPUT_DIR", "data/testing-output-run/navier_stokes_output")
FIELDS_DIR = os.path.join(OUTPUT_DIR, "fields")
REL_TOL = 1e-6
ABS_TOL = 1e-9


def find_latest_step_file():
    if not os.path.isdir(FIELDS_DIR):
        return None
    step_files = [
        f for f in os.listdir(FIELDS_DIR) if re.match(r"step_\d+\.json$", f)
    ]
    if not step_files:
        return None
    step_numbers = [int(re.search(r"(\d+)", f).group(1)) for f in step_files]
    max_step = max(step_numbers)
    return f"step_{max_step:04d}.json"


@pytest.mark.skipif(
    not os.path.isdir(FIELDS_DIR),
    reason="Fields directory not found"
)
def test_cavity_centerline_u_profile_matches_reference():
    latest_step_file = find_latest_step_file()
    if not latest_step_file:
        pytest.skip("No step_*.json files found in fields/")

    path = os.path.join(FIELDS_DIR, latest_step_file)
    with open(path, "r") as f:
        data = json.load(f)

    velocity = np.array(data["velocity"])
    grid_shape = velocity.shape[:-1]

    if len(grid_shape) == 2:
        Nx, Ny = grid_shape
        center_x = Nx // 2
        u_profile = velocity[center_x, :, 0]
        y_coords = np.linspace(0, 1, Ny)
    else:
        Nx, Ny, Nz = grid_shape
        center_x = Nx // 2
        center_z = Nz // 2
        u_profile = velocity[center_x, :, center_z, 0]
        y_coords = np.linspace(0, 1, Ny)

    ref_y = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    ref_u = np.array([0.0, -0.22, -0.38, -0.47, -0.38, -0.22, 0.09, 1.0])

    u_interp = np.interp(ref_y, y_coords, u_profile)

    assert np.allclose(u_interp, ref_u, rtol=REL_TOL, atol=ABS_TOL), \
        f"Velocity profile in {latest_step_file} deviates from reference at cavity centerline"



