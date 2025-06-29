import numpy as np
import pytest
from unittest.mock import patch
from src.numerical_methods.implicit_solver import ImplicitSolver

@pytest.fixture
def mesh_and_props():
    mesh_info = {
        "grid_shape": (5, 5, 5),
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "boundary_conditions": {}
    }
    fluid_props = {"density": 1.0, "viscosity": 0.1}
    return mesh_info, fluid_props

@patch("src.numerical_methods.implicit_solver.compute_advection_term", return_value=np.ones((5, 5, 5)))
@patch("src.numerical_methods.implicit_solver.compute_diffusion_term", return_value=np.ones((5, 5, 5)))
@patch("src.numerical_methods.implicit_solver.compute_pressure_divergence", return_value=np.zeros((5, 5, 5)))
@patch("src.numerical_methods.implicit_solver.solve_poisson_for_phi", return_value=np.zeros((5, 5, 5)))
@patch("src.numerical_methods.implicit_solver.apply_pressure_correction", side_effect=lambda u, p, phi, m, dt, rho: (u * 0.5, p + 0.1))
@patch("src.numerical_methods.implicit_solver.apply_boundary_conditions", side_effect=lambda u, p, f, m, is_tentative_step: (u, p))
def test_implicit_solver_step_mocked(
    mock_boundary,
    mock_pcorr,
    mock_poisson,
    mock_divergence,
    mock_diffusion,
    mock_advection,
    mesh_and_props
):
    velocity = np.ones((5, 5, 5, 3)) * 0.2
    pressure = np.zeros((5, 5, 5))
    mesh_info, fluid_props = mesh_and_props
    solver = ImplicitSolver(fluid_props, mesh_info, dt=0.1)

    u_out, p_out = solver.step(velocity, pressure)

    assert u_out.shape == velocity.shape
    assert p_out.shape == pressure.shape
    assert not np.allclose(u_out, velocity), "Velocity should evolve over iterations"
    assert np.allclose(p_out, 0.5), "Pressure should accumulate mock offsets (5 iterations × 0.1)"

    # Ensure mocks were called multiple times due to pseudo-loop
    assert mock_advection.call_count == 15  # 3 components × 5 iterations
    assert mock_diffusion.call_count == 15
    assert mock_divergence.call_count == 5
    assert mock_poisson.call_count == 5
    assert mock_pcorr.call_count == 5
    assert mock_boundary.call_count == 6  # 5 loops + final BC



