# tests/test_solver_core/test_poisson_functional.py

import numpy as np
from src.numerical_methods.poisson_solver import solve_poisson_for_phi
from tests.test_solver_core.test_utils import create_mesh_info, add_zero_padding


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
    assert error < 0.025


def test_poisson_output_matches_rhs_shape():
    nx, ny, nz = 8, 10, 12
    rhs_core = np.random.randn(nx, ny, nz)
    rhs_padded = add_zero_padding(rhs_core)
    mesh = create_mesh_info(nx, ny, nz)
    phi = solve_poisson_for_phi(rhs_padded, mesh, time_step=1.0, max_iterations=200)
    assert phi.shape == rhs_padded.shape


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



