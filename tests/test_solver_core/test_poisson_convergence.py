# tests/test_solver_core/test_poisson_convergence.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from tests.test_solver_core.test_utils import create_mesh_info, add_zero_padding


def test_poisson_convergence_improves_with_iterations():
    nx, ny, nz = 10, 10, 10
    dx = 1.0 / nx
    mesh = create_mesh_info(nx, ny, nz, dx, dx, dx)
    rhs_core = -6.0 * np.ones((nx, ny, nz))
    rhs_padded = add_zero_padding(rhs_core)

    phi_low = solve_poisson_for_phi(rhs_padded.copy(), mesh, 1.0, max_iterations=50)
    phi_high = solve_poisson_for_phi(rhs_padded.copy(), mesh, 1.0, max_iterations=500)

    diff_low = np.mean(np.abs(phi_low[1:-1, 1:-1, 1:-1] - np.mean(phi_low)))
    diff_high = np.mean(np.abs(phi_high[1:-1, 1:-1, 1:-1] - np.mean(phi_high)))

    assert diff_high < diff_low


def test_poisson_converges_toward_zero_residual_when_rhs_zero():
    nx, ny, nz = 8, 8, 8
    rhs_core = np.zeros((nx, ny, nz))
    rhs_padded = add_zero_padding(rhs_core)
    mesh = create_mesh_info(nx, ny, nz)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=500)

    laplacian = (
        -6.0 * phi[1:-1, 1:-1, 1:-1] +
        phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
        phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
        phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
    )
    residual = np.linalg.norm(laplacian)
    assert residual < 1e-6


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
    nx, ny, nz = 12, 12, 12
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

    laplacian = (
        -6.0 * phi[1:-1, 1:-1, 1:-1] +
        phi[2:, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
        phi[1:-1, 2:, 1:-1] + phi[1:-1, :-2, 1:-1] +
        phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
    )
    residual = np.linalg.norm(laplacian - rhs_core)
    assert residual < 1e-4



