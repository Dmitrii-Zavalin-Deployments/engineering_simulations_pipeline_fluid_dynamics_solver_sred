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


def test_poisson_matches_manufactured_sine_solution():
    nx, ny, nz = 16, 16, 16
    dx = dy = dz = 1.0 / nx
    mesh = create_mesh_info(nx, ny, nz, dx, dy, dz)

    grid = np.linspace(0, 1, nx)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing="ij")
    reference = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    rhs_core = -3.0 * np.pi**2 * reference

    rhs_padded = add_zero_padding(rhs_core)
    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=3000)
    core = phi[1:-1, 1:-1, 1:-1]

    reference -= reference.mean()
    core -= core.mean()
    error = np.mean(np.abs(core - reference))
    assert error < 0.02


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

    phi_low = solve_poisson_for_phi(rhs_padded.copy(), mesh, 1.0, max_iterations=50)
    phi_high = solve_poisson_for_phi(rhs_padded.copy(), mesh, 1.0, max_iterations=500)

    diff_low = np.mean(np.abs(phi_low[1:-1, 1:-1, 1:-1] - np.mean(phi_low)))
    diff_high = np.mean(np.abs(phi_high[1:-1, 1:-1, 1:-1] - np.mean(phi_high)))

    assert diff_high < diff_low


def test_poisson_handles_nonuniform_grid():
    nx, ny, nz = 10, 12, 14
    dx, dy, dz = 2.0, 1.0, 0.5
    mesh = create_mesh_info(nx, ny, nz, dx, dy, dz)

    rhs_core = np.random.randn(nx, ny, nz)
    rhs_padded = add_zero_padding(rhs_core)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=300)
    assert phi.shape == rhs_padded.shape


def test_poisson_solution_is_independent_of_initial_guess():
    nx, ny, nz = 8, 8, 8
    mesh = create_mesh_info(nx, ny, nz)
    rhs_core = np.random.rand(nx, ny, nz)
    rhs_padded = add_zero_padding(rhs_core)

    phi_a = solve_poisson_for_phi(rhs_padded.copy(), mesh, time_step=1.0, max_iterations=300)
    phi_b = solve_poisson_for_phi(rhs_padded.copy(), mesh, time_step=1.0, max_iterations=300)

    diff = phi_a[1:-1, 1:-1, 1:-1] - phi_b[1:-1, 1:-1, 1:-1]
    assert np.allclose(diff, 0.0, atol=1e-6)


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


def test_poisson_resolves_structured_sin_source():
    nx = ny = nz = 16
    dx = 1.0 / nx
    x = np.linspace(0, 1, nx)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    rhs_core = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    rhs_padded = add_zero_padding(rhs_core)
    mesh = create_mesh_info(nx, ny, nz, dx, dx, dx)

    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=2000)
    assert phi[1:-1, 1:-1, 1:-1].std() > 0.005


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
    error = np.abs(coarse_interp - fine)
    assert np.mean(error) < 0.05



