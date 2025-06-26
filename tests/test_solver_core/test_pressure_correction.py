# tests/test_solver_core/test_pressure_correction.py

import numpy as np
import pytest
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import (
    calculate_gradient,
    apply_pressure_correction,
)


def mesh_metadata(shape, dx=1.0, dy=1.0, dz=1.0):
    return {
        "grid_shape": (shape[0] + 2, shape[1] + 2, shape[2] + 2),
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def test_calculate_gradient_matches_analytic_derivative():
    nx, ny, nz = 8, 8, 8
    dx = 1.0 / nx
    x = np.linspace(0, 1, nx)
    X = np.broadcast_to(x[:, None, None], (nx, ny, nz))
    field = X**2
    grad_numeric = calculate_gradient(field, dx, axis=0)
    grad_exact = 2 * X
    error = np.abs(grad_numeric - grad_exact).mean()
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

    phi_core = np.ones((nx, ny, nz))
    p_field = np.zeros((nx + 2, ny + 2, nz + 2))
    velocity = np.zeros((nx + 2, ny + 2, nz + 2, 3))

    corrected_u, updated_p = apply_pressure_correction(velocity, p_field, phi_core, mesh, dt, rho)

    assert np.allclose(updated_p[1:-1, 1:-1, 1:-1], phi_core)
    assert np.any(corrected_u[1:-1, 1:-1, 1:-1, :] != 0.0)
    assert np.all(corrected_u[0, :, :, :] == 0.0)
    assert np.all(corrected_u[:, 0, :, :] == 0.0)
    assert np.all(corrected_u[:, :, 0, :] == 0.0)


def test_velocity_correction_scales_with_density():
    shape = (6, 6, 6)
    phi = np.ones(shape)
    velocity = np.zeros((8, 8, 8, 3))
    pressure = np.zeros((8, 8, 8))
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    dt = 1.0

    vel_low_rho, _ = apply_pressure_correction(velocity.copy(), pressure.copy(), phi, mesh, dt, density=0.5)
    vel_high_rho, _ = apply_pressure_correction(velocity.copy(), pressure.copy(), phi, mesh, dt, density=2.0)

    vmag_low = np.linalg.norm(vel_low_rho[1:-1, 1:-1, 1:-1], axis=-1).mean()
    vmag_high = np.linalg.norm(vel_high_rho[1:-1, 1:-1, 1:-1], axis=-1).mean()

    assert vmag_low > 3.5 * vmag_high


def test_pressure_correction_reduces_divergence():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    x = np.linspace(0, 1, shape[0] + 2)
    y = np.linspace(0, 1, shape[1] + 2)
    z = np.linspace(0, 1, shape[2] + 2)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    velocity = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
    velocity[..., 0] = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.cos(np.pi * Z)
    velocity[..., 1] = np.cos(np.pi * X) * np.sin(np.pi * Y) * np.cos(np.pi * Z)
    velocity[..., 2] = np.cos(np.pi * X) * np.cos(np.pi * Y) * np.sin(np.pi * Z)

    div_before = compute_pressure_divergence(velocity, mesh)
    rhs = div_before.copy()
    padded_rhs = np.pad(rhs, ((1, 1), (1, 1), (1, 1)), mode="constant")

    phi = solve_poisson_for_phi(padded_rhs, mesh, time_step=1.0, max_iterations=3000)

    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1], mesh, time_step=1.0, density=1.0
    )

    div_after = compute_pressure_divergence(corrected_u, mesh)
    assert np.mean(np.abs(div_after)) < 0.1 * np.mean(np.abs(div_before))


def test_pressure_projection_converges_over_multiple_steps():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    velocity = np.random.randn(shape[0] + 2, shape[1] + 2, shape[2] + 2, 3) * 0.1

    for _ in range(5):
        rhs = compute_pressure_divergence(velocity, mesh)
        padded_rhs = np.pad(rhs, ((1, 1), (1, 1), (1, 1)), mode="constant")
        phi = solve_poisson_for_phi(padded_rhs, mesh, time_step=1.0, max_iterations=1000)
        velocity, _ = apply_pressure_correction(
            velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1], mesh, 1.0, 1.0
        )

    final_div = compute_pressure_divergence(velocity, mesh)
    assert np.mean(np.abs(final_div)) < 1e-5


def test_boundary_sensitive_pressure_gradient():
    nx = ny = nz = 8
    dx = 1.0 / nx
    mesh = mesh_metadata((nx, ny, nz), dx, dx, dx)
    phi = np.zeros((nx, ny, nz))
    phi[-1, :, :] = 10.0  # Sharp gradient near boundary

    velocity = np.zeros((nx + 2, ny + 2, nz + 2, 3))
    pressure = np.zeros((nx + 2, ny + 2, nz + 2))

    corrected_u, _ = apply_pressure_correction(velocity.copy(), pressure.copy(), phi, mesh, time_step=1.0, density=1.0)

    corner_velocity = corrected_u[-2, ny//2, nz//2, 0]
    assert corner_velocity < -4.0  # Expect strong correction at boundary-facing cell


def test_kinetic_energy_drops_after_projection():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    velocity = np.random.randn(shape[0] + 2, shape[1] + 2, shape[2] + 2, 3)

    ke_before = 0.5 * np.sum(velocity[1:-1, 1:-1, 1:-1]**2)

    rhs = compute_pressure_divergence(velocity, mesh)
    padded_rhs = np.pad(rhs, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(padded_rhs, mesh, time_step=1.0, max_iterations=1000)

    corrected_u, _ = apply_pressure_correction(velocity, np.zeros_like(phi), phi[1:-1, 1:-1, 1:-1], mesh, 1.0, 1.0)
    ke_after = 0.5 * np.sum(corrected_u[1:-1, 1:-1, 1:-1]**2)

    assert ke_after < ke_before



