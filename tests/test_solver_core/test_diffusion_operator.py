# tests/test_solver_core/test_diffusion_operator.py

import numpy as np
import pytest
from src.numerical_methods.diffusion import apply_diffusion_step

def create_test_field(nx, ny, nz, value=1.0):
    """
    Returns a padded 3D field with uniform interior values and ghost cells = 0.
    """
    padded_shape = (nx + 2, ny + 2, nz + 2)
    field = np.zeros(padded_shape, dtype=np.float64)
    field[1:-1, 1:-1, 1:-1] = value
    return field

def test_diffusion_does_not_change_uniform_field():
    """
    A uniform field should remain unchanged under diffusion.
    """
    nx, ny, nz = 8, 8, 8
    dt = 0.01
    nu = 0.1

    mesh_info = {
        "grid_shape": (nx + 2, ny + 2, nz + 2),
        "dx": 1.0, "dy": 1.0, "dz": 1.0
    }

    u = create_test_field(nx, ny, nz, value=5.0)

    u_new = apply_diffusion_step(u.copy(), diffusion_coefficient=nu, mesh_info=mesh_info, dt=dt)

    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    assert np.allclose(u_new[interior], u[interior], atol=1e-10), "Uniform field should remain constant"

def test_diffusion_smooths_sharp_peak():
    """
    A central peak should smooth over time due to diffusion.
    """
    nx, ny, nz = 5, 5, 5
    dt = 0.01
    nu = 0.1

    mesh_info = {
        "grid_shape": (nx + 2, ny + 2, nz + 2),
        "dx": 1.0, "dy": 1.0, "dz": 1.0
    }

    u = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)
    u[3, 3, 3] = 10.0  # Central spike

    u_new = apply_diffusion_step(u.copy(), diffusion_coefficient=nu, mesh_info=mesh_info, dt=dt)

    peak_before = u[3, 3, 3]
    peak_after = u_new[3, 3, 3]
    neighbors = [
        u_new[4, 3, 3], u_new[2, 3, 3],
        u_new[3, 4, 3], u_new[3, 2, 3],
        u_new[3, 3, 4], u_new[3, 3, 2]
    ]

    assert peak_after < peak_before, "Central value should decrease due to diffusion"
    assert all(n > 0 for n in neighbors), "Neighbors should increase due to smoothing"



