# tests/test_solver_core/test_poisson_convergence.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from tests.test_solver_core.test_utils import create_mesh_info, add_zero_padding


def compute_scaled_laplacian(phi, dx):
    return (
        -6.0 * phi[1:-1, 1:-1, 1:-1] +
        phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
        phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
        phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
    ) / dx**2


def test_poisson_convergence_improves_with_iterations():
    nx, ny, nz = 10, 10, 10
    dx = 1.0 / nx
    mesh = create_mesh_info(nx, ny, nz, dx, dx, dx)
    rhs_core = -6.0 * np.ones((nx, ny, nz))
    rhs_padded = add_zero_padding(rhs_core)

    phi_low = solve_poisson_for_phi(rhs_padded.copy(), mesh, time_step=1.0, max_iterations=20)
    phi_high = solve_poisson_for_phi(rhs_padded.copy(), mesh, time_step=1.0, max_iterations=200)

    lap_low = compute_scaled_laplacian(phi_low, dx)
    lap_high = compute_scaled_laplacian(phi_high, dx)

    res_low = np.linalg.norm(lap_low - rhs_core)
    res_high = np.linalg.norm(lap_high - rhs_core)

    # Allow for equality within floating-point tolerance
    assert res_high < res_low or np.isclose(res_high, res_low, atol=1e-6)


def test_poisson_converges_toward_zero_residual_when_rhs_zero():
    nx, ny, nz = 8, 8, 8
    dx = 1.0 / nx
    rhs_core = np.zeros((nx, ny, nz))
    rhs_padded = add_zero_padding(rhs_core)
    mesh = create_mesh_info(nx, ny, nz, dx, dx, dx)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=500)
    laplacian = compute_scaled_laplacian(phi, dx)

    assert np.linalg.norm(laplacian) < 1e-6


def test_poisson_solution_converges_with_grid_refinement():
    def run_solver(n):
        dx = 1.0 / n
        mesh = create_mesh_info(n, n, n, dx, dx, dx)
        rhs_core = -6.0 * np.ones((n, n, n))
        rhs_padded = add_zero_padding(rhs_core)
        phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=2000)
        return phi[1:-1, 1:-1, 1:-1]

    coarse = run_solver(8)
    fine = run_solver(16)

    from scipy.ndimage import zoom
    coarse_interp = zoom(coarse, 2, order=1)
    fine = fine[:coarse_interp.shape[0], :coarse_interp.shape[1], :coarse_interp.shape[2]]

    coarse_interp -= coarse_interp.mean()
    fine -= fine.mean()
    error = np.mean(np.abs(coarse_interp - fine))

    assert error < 0.05


def test_poisson_residual_decay_against_strict_tolerance():
    nx = ny = nz = 12
    dx = 1.0 / nx
    mesh = create_mesh_info(nx, ny, nz, dx, dx, dx)

    X, Y, Z = np.meshgrid(
        np.linspace(0, 1, nx),
        np.linspace(0, 1, ny),
        np.linspace(0, 1, nz),
        indexing="ij"
    )
    rhs_core = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    rhs_padded = add_zero_padding(rhs_core)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=10000, tolerance=1e-8)
    laplacian = compute_scaled_laplacian(phi, dx)

    residual = np.linalg.norm(laplacian - rhs_core)
    rel_residual = residual / np.linalg.norm(rhs_core)

    assert rel_residual < 1e-3



