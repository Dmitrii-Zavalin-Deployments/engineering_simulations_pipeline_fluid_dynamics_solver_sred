# tests/test_solver_core/test_pressure_projection.py

import numpy as np
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from src.numerical_methods.pressure_correction import apply_pressure_correction
from tests.test_solver_core.test_utils import mesh_metadata


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
    padded_rhs = np.pad(div_before, ((1, 1), (1, 1), (1, 1)), mode="constant")

    phi = solve_poisson_for_phi(
        padded_rhs, mesh, time_step=1.0, max_iterations=1000, tolerance=1e-6
    )

    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi),
        phi[1:-1, 1:-1, 1:-1],
        mesh, time_step=1.0, density=1.0
    )

    div_after = compute_pressure_divergence(corrected_u, mesh)
    reduction_ratio = np.mean(np.abs(div_after)) / np.mean(np.abs(div_before))

    # This threshold reflects realistic expectations from a single SOR-based correction step.
    assert reduction_ratio < 0.6, (
        f"Divergence only reduced by {reduction_ratio:.3f}, expected less than 0.6"
    )


def test_pressure_projection_converges_over_multiple_steps():
    shape = (16, 16, 16)
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    velocity = np.random.randn(shape[0] + 2, shape[1] + 2, shape[2] + 2, 3) * 0.1

    div_history = []
    for _ in range(6):
        divergence = compute_pressure_divergence(velocity, mesh)
        div_history.append(np.mean(np.abs(divergence)))

        padded_rhs = np.pad(divergence, ((1, 1), (1, 1), (1, 1)), mode="constant")
        phi = solve_poisson_for_phi(
            padded_rhs, mesh, time_step=1.0, max_iterations=1000, tolerance=1e-6
        )
        velocity, _ = apply_pressure_correction(
            velocity, np.zeros_like(phi),
            phi[1:-1, 1:-1, 1:-1],
            mesh, 1.0, 1.0
        )

    reduction_ratio = div_history[-1] / div_history[0]

    # Expect gradual convergence; SOR is iterative and not exact over a few steps.
    assert reduction_ratio < 0.4, (
        f"Final divergence ratio was {reduction_ratio:.3f}, expected less than 0.4"
    )


def test_boundary_sensitive_pressure_gradient():
    nx = ny = nz = 8
    dx = 1.0 / nx
    mesh = mesh_metadata((nx, ny, nz), dx, dx, dx)

    phi = np.zeros((nx, ny, nz))
    phi[-1, :, :] = 10.0  # Induce sharp boundary gradient

    velocity = np.zeros((nx + 2, ny + 2, nz + 2, 3))
    pressure = np.zeros((nx + 2, ny + 2, nz + 2))

    corrected_u, _ = apply_pressure_correction(
        velocity, pressure, phi, mesh, time_step=1.0, density=1.0
    )

    corner_velocity = corrected_u[-2, ny // 2, nz // 2, 0]
    assert corner_velocity < -4.0  # Strong leftward velocity expected


def test_kinetic_energy_drops_after_projection():
    shape = (10, 10, 10)
    mesh = mesh_metadata(shape, dx=1.0 / shape[0])
    velocity = np.random.randn(shape[0] + 2, shape[1] + 2, shape[2] + 2, 3)

    ke_before = 0.5 * np.sum(velocity[1:-1, 1:-1, 1:-1] ** 2)

    rhs = compute_pressure_divergence(velocity, mesh)
    padded_rhs = np.pad(rhs, ((1, 1), (1, 1), (1, 1)), mode="constant")
    phi = solve_poisson_for_phi(
        padded_rhs, mesh, time_step=1.0, max_iterations=1000, tolerance=1e-6
    )

    corrected_u, _ = apply_pressure_correction(
        velocity, np.zeros_like(phi),
        phi[1:-1, 1:-1, 1:-1],
        mesh, 1.0, 1.0
    )
    ke_after = 0.5 * np.sum(corrected_u[1:-1, 1:-1, 1:-1] ** 2)

    assert ke_after < ke_before




