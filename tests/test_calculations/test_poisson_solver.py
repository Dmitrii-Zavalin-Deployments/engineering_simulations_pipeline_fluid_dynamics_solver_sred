import numpy as np
import pytest
from src.numerical_methods.poisson_solver import solve_poisson_for_phi

@pytest.fixture
def mesh_info():
    return {"grid_shape": (5, 5, 5), "dx": 1.0, "dy": 1.0, "dz": 1.0}

def test_poisson_solver_returns_correct_shape(mesh_info):
    divergence = np.ones((5, 5, 5))
    phi = solve_poisson_for_phi(divergence, mesh_info, time_step=0.1)
    assert phi.shape == divergence.shape
    assert np.isfinite(phi).all()

def test_poisson_solver_residual_reporting(mesh_info):
    divergence = np.ones((5, 5, 5))
    phi, residual = solve_poisson_for_phi(divergence, mesh_info, time_step=0.1, return_residual=True)
    assert isinstance(residual, float)
    assert residual >= 0
    assert phi.shape == divergence.shape

def test_poisson_solver_convergence_on_zero_rhs(mesh_info):
    divergence = np.zeros((5, 5, 5))  # φ should stay zero if RHS is zero
    phi, residual = solve_poisson_for_phi(divergence, mesh_info, time_step=0.1,
                                          return_residual=True, max_iterations=100, tolerance=1e-10)
    assert np.allclose(phi, 0.0, atol=1e-10)
    assert residual < 1e-10

def test_poisson_solver_raises_for_unsupported_backend(mesh_info):
    divergence = np.ones((5, 5, 5))
    with pytest.raises(ValueError, match="Unsupported backend"):
        solve_poisson_for_phi(divergence, mesh_info, time_step=0.1, backend="invalid_method")

def test_poisson_solver_matches_analytic_solution(mesh_info):
    """
    Verifies that the Poisson solver recovers the known φ(x,y,z) = sin(πx)sin(πy)sin(πz)
    by feeding it a manufactured source term: b = -∇²φ.
    """
    nx, ny, nz = mesh_info["grid_shape"]
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    dt = 0.1

    # Physical domain: [0, 1]
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Known solution and source term
    true_phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    laplace_phi = -3 * (np.pi ** 2) * true_phi
    divergence = laplace_phi * dt  # RHS for solve_poisson_for_phi expects divergence

    recovered_phi, residual = solve_poisson_for_phi(
        divergence, mesh_info, time_step=dt,
        tolerance=1e-8, max_iterations=2000,
        return_residual=True
    )

    error = np.abs(recovered_phi - true_phi)
    max_error = np.max(error)
    assert max_error < 1e-2, f"Max error too high: {max_error}"



