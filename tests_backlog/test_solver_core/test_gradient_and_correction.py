# tests/test_solver_core/test_gradient_and_correction.py

import numpy as np
import pytest
from src.numerical_methods.pressure_correction import (
    calculate_gradient,
    apply_pressure_correction,
)
from tests.test_solver_core.test_utils import mesh_metadata


def test_calculate_gradient_matches_analytic_derivative():
    nx, ny, nz = 64, 8, 8
    dx = 1.0 / nx
    x = np.linspace(0, 1, nx)
    X = np.broadcast_to(x[:, None, None], (nx, ny, nz))
    field = X**2
    grad_numeric = calculate_gradient(field, dx, axis=0)
    grad_exact = 2 * X
    error = np.abs(grad_numeric[2:-2] - grad_exact[2:-2]).mean()
    assert error < 0.02


def test_calculate_gradient_raises_on_invalid_axis():
    field = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="Axis must be 0, 1, or 2."):
        _ = calculate_gradient(field, h=1.0, axis=3)


def test_apply_pressure_correction_modifies_velocity_and_pressure():
    nx, ny, nz = 6, 6, 6
    dx = dy = dz = 1.0 / nx
    mesh = mesh_metadata((nx, ny, nz), dx, dy, dz)
    dt = 0.1
    rho = 1.0

    x = np.linspace(0, 1, nx)
    X = np.broadcast_to(x[:, None, None], (nx, ny, nz))
    phi_core = X

    p_field = np.zeros((nx + 2, ny + 2, nz + 2))
    velocity = np.zeros((nx + 2, ny + 2, nz + 2, 3))

    corrected_u, updated_p = apply_pressure_correction(
        velocity, p_field, phi_core, mesh, dt, rho
    )

    assert np.allclose(updated_p[1:-1, 1:-1, 1:-1], phi_core)
    assert np.any(corrected_u[1:-1, 1:-1, 1:-1, :] != 0.0)
    assert np.all(corrected_u[0, :, :, :] == 0.0)
    assert np.all(corrected_u[:, 0, :, :] == 0.0)
    assert np.all(corrected_u[:, :, 0, :] == 0.0)


def test_velocity_correction_scales_with_density():
    shape = (6, 6, 6)
    x = np.linspace(0, 1, shape[0])
    X = np.broadcast_to(x[:, None, None], shape)
    phi = X

    velocity = np.zeros((8, 8, 8, 3))
    pressure = np.zeros((8, 8, 8))
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    dt = 1.0

    vel_low_rho, _ = apply_pressure_correction(
        velocity.copy(), pressure.copy(), phi, mesh, dt, density=0.5
    )
    vel_high_rho, _ = apply_pressure_correction(
        velocity.copy(), pressure.copy(), phi, mesh, dt, density=2.0
    )

    vmag_low = np.linalg.norm(
        vel_low_rho[1:-1, 1:-1, 1:-1], axis=-1
    ).mean()
    vmag_high = np.linalg.norm(
        vel_high_rho[1:-1, 1:-1, 1:-1], axis=-1
    ).mean()

    assert vmag_low > 3.5 * vmag_high



