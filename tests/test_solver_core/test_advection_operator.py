# tests/test_solver_core/test_advection_operator.py

import numpy as np
import pytest

# 🔧 Update this import path to match your actual module structure
from src.numerical_methods.advection import advect_velocity

def test_uniform_velocity_field_should_remain_unchanged():
    """
    For a uniform velocity field, the advection term should vanish.
    This tests that the real advection function preserves uniformity (i.e., zero self-advection).
    """
    nx, ny, nz = 6, 6, 6
    dx = dy = dz = 1.0
    dt = 0.1

    # Add ghost padding
    shape = (nx + 2, ny + 2, nz + 2)
    u = np.ones(shape) * 1.0
    v = np.ones(shape) * 1.0
    w = np.ones(shape) * 1.0

    u_new, v_new, w_new = advect_velocity(u, v, w, dx, dy, dz, dt)

    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    assert np.allclose(u_new, u[interior]), "u field changed unexpectedly under uniform flow"
    assert np.allclose(v_new, v[interior]), "v field changed unexpectedly under uniform flow"
    assert np.allclose(w_new, w[interior]), "w field changed unexpectedly under uniform flow"



