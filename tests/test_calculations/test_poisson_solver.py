import numpy as np
import pytest
from src.numerical_methods.poisson_solver import solve_poisson_for_phi

@pytest.fixture
def mesh_info():
    N = 31  # Increased resolution for accurate sinusoidal modeling
    h = 1.0 / (N - 1)  # For domain [0, 1], spacing = 1 / (N - 1)
    return {
        "grid_shape": (N, N, N),
        "dx": h,
        "dy": h,
        "dz": h
    }

def test_poisson_solver_returns_correct_shape(mesh_info):
    shape = mesh_info["grid_shape"]
    divergence = np.ones(shape)
    phi = solve_poisson_for_phi(divergence, mesh_info, time_step=0.1)
    assert phi.shape == divergence.shape
    assert np.isfinite(phi).all()

def test_poisson_solver_residual_reporting(mesh_info):
    shape = mesh_info["grid_shape"]
    divergence = np.ones(shape)
    phi, residual = solve_poisson_for_phi(divergence, mesh_info, time_step=0.1, return_residual=True)
    assert isinstance(residual, float)
    assert residual >= 0
    assert phi.shape == divergence.shape

def test_poisson_solver_convergence_on_zero_rhs(mesh_info):
    shape = mesh_info["grid_shape"]
    divergence = np.zeros(shape)
    phi, residual = solve_poisson_for_phi(divergence, mesh_info, time_step=0.1,
                                          return_residual=True, max_iterations=200, tolerance=1e-10)
    assert np.allclose(phi, 0.0, atol=1e-10)
    assert residual < 1e-10

def test_poisson_solver_raises_for_unsupported_backend(mesh_info):
    shape = mesh_info["grid_shape"]
    divergence = np.ones(shape)
    with pytest.raises(ValueError, match="Unsupported backend"):
        solve_poisson_for_phi(divergence, mesh_info, time_step=0.1, backend="invalid_method")

def test_poisson_solver_matches_analytic_solution(mesh_info):
    """
    Tests whether the Poisson solver can accurately recover a known solution
    φ(x,y,z) = sin(πx)sin(πy)sin(πz) by solving -∇²φ = b.
    """
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    dt = 0.1

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Known solution and RHS
    true_phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    laplace_phi = -3 * (np.pi ** 2) * true_phi
    divergence = laplace_phi * dt

    recovered_phi, residual = solve_poisson_for_phi(
        divergence, mesh_info, time_step=dt,
        tolerance=1e-10, max_iterations=3000,
        return_residual=True
    )

    # Compare interior values only to avoid boundary condition mismatch
    error = np.abs(recovered_phi[1:-1, 1:-1, 1:-1] - true_phi[1:-1, 1:-1, 1:-1])
    max_error = np.max(error)
    assert max_error < 1e-2, f"Max error too high: {max_error}"



