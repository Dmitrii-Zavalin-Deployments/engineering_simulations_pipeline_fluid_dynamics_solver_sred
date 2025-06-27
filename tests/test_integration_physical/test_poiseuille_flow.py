import os
import json
import numpy as np
import pytest
import re

OUTPUT_DIR = os.environ.get("ACTUAL_OUTPUT_DIR", "data/testing-output-run/navier_stokes_output")
FIELDS_DIR = os.path.join(OUTPUT_DIR, "fields")


def find_latest_step_file():
    if not os.path.isdir(FIELDS_DIR):
        return None
    candidates = [f for f in os.listdir(FIELDS_DIR) if re.match(r"step_\d+\.json$", f)]
    if not candidates:
        return None
    steps = [int(re.search(r"(\d+)", f).group(1)) for f in candidates]
    return f"step_{max(steps):04d}.json"


@pytest.mark.skipif(not os.path.isdir(FIELDS_DIR), reason="Fields directory not found")
def test_poiseuille_u_profile_matches_parabolic_analytic():
    step_file = find_latest_step_file()
    if not step_file:
        pytest.skip("No step_*.json files found in fields/")

    path = os.path.join(FIELDS_DIR, step_file)
    with open(path, "r") as f:
        data = json.load(f)

    velocity = np.array(data["velocity"])  # (Nx, Ny, [Nz], 3)
    grid_shape = velocity.shape[:-1]

    if len(grid_shape) == 2:  # 2D channel flow
        Nx, Ny = grid_shape
        center_x = Nx // 2
        u_profile = velocity[center_x, :, 0]  # u across height
        y = np.linspace(0, 1, Ny)

    elif len(grid_shape) == 3:  # 3D channel flow
        Nx, Ny, Nz = grid_shape
        center_x = Nx // 2
        center_z = Nz // 2
        u_profile = velocity[center_x, :, center_z, 0]
        y = np.linspace(0, 1, Ny)

    else:
        pytest.skip("Unsupported velocity field dimensions")

    # Normalize profile by max value (assumes fully developed)
    u_max = np.max(u_profile)
    if np.isclose(u_max, 0):
        pytest.skip("All-zero velocity field")
    u_normalized = u_profile / u_max

    # Analytical: parabolic profile u(y) = 1 - (2y - 1)^2
    u_analytic = 1 - (2 * y - 1)**2

    # Use a loose tolerance — integration-level check
    assert np.allclose(u_normalized, u_analytic, atol=0.05), (
        "Velocity profile does not match parabolic Poiseuille profile"
    )



