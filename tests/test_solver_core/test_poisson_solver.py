# tests/test_solver_core/test_poisson_solver.py

import numpy as np
import pytest
from src.numerical_methods.poisson_solver import solve_poisson_for_phi


def create_mesh_info(nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    return {
        "grid_shape": (nx + 2, ny + 2, nz + 2),
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }


def add_zero_padding(core_field):
    padded = np.zeros(tuple(s + 2 for s in core_field.shape))
    padded[1:-1, 1:-1, 1:-1] = core_field
    return padded


def test_poisson_zero_rhs_returns_flat_solution():
    nx, ny, nz = 6, 6, 6
    rhs = np.zeros((nx, ny, nz))
    rhs_padded = add_zero_padding(rhs)
    mesh = create_mesh_info(nx, ny, nz)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=500)
    core = phi[1:-1, 1:-1, 1:-1]
    assert np.allclose(core - np.mean(core), 0.0, atol=1e-10)


def test_poisson_matches_known_quadratic_solution():
    nx, ny, nz = 12, 12, 12
    dx = dy = dz = 1.0 / nx
    mesh = create_mesh_info(nx, ny, nz, dx, dy, dz)

    rhs_core = -6.0 * np.ones((nx, ny, nz))  # ∇²(x² + y² + z²) = -6
    rhs_padded = add_zero_padding(rhs_core)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=10000)
    core = phi[1:-1, 1:-1, 1:-1]

    grid = np.linspace(0, 1, nx)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    reference = X**2 + Y**2 + Z**2
    reference -= reference.mean()
    core -= core.mean()

    assert np.allclose(core, reference, atol=1e-2)


def test_poisson_output_matches_rhs_shape():
    nx, ny, nz = 8, 10, 12
    rhs_core = np.random.randn(nx, ny, nz)
    rhs_padded = add_zero_padding(rhs_core)
    mesh = create_mesh_info(nx, ny, nz)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=200)
    assert phi.shape == rhs_padded.shape


def test_poisson_convergence_improves_with_iterations():
    nx, ny, nz = 10, 10, 10
    dx = 1.0 / nx
    mesh = create_mesh_info(nx, ny, nz, dx, dx, dx)

    rhs_core = -6.0 * np.ones((nx, ny, nz))
    rhs_padded = add_zero_padding(rhs_core)

    phi_low_iter = solve_poisson_for_phi(rhs_padded.copy(), mesh, time_step=1.0, max_iterations=50)
    phi_high_iter = solve_poisson_for_phi(rhs_padded.copy(), mesh, time_step=1.0, max_iterations=500)

    diff_low = np.mean(np.abs(phi_low_iter[1:-1, 1:-1, 1:-1] - np.mean(phi_low_iter)))
    diff_high = np.mean(np.abs(phi_high_iter[1:-1, 1:-1, 1:-1] - np.mean(phi_high_iter)))

    assert diff_high < diff_low


def test_poisson_handles_nonuniform_grid():
    nx, ny, nz = 10, 12, 14
    dx, dy, dz = 2.0, 1.0, 0.5
    mesh = create_mesh_info(nx, ny, nz, dx, dy, dz)

    rhs_core = np.random.randn(nx, ny, nz)
    rhs_padded = add_zero_padding(rhs_core)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=300)
    assert phi.shape == rhs_padded.shape



