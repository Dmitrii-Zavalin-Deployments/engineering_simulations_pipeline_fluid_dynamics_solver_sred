import numpy as np
import pytest
from unittest.mock import patch
from src.numerical_methods.explicit_solver import ExplicitSolver

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

@pytest.mark.parametrize("initial_velocity", [
    np.zeros((5, 5, 5, 3)),
    np.ones((5, 5, 5, 3)) * 0.5
])
@patch("src.numerical_methods.explicit_solver.compute_advection_term", return_value=np.ones((5, 5, 5)))
@patch("src.numerical_methods.explicit_solver.compute_diffusion_term", return_value=np.ones((5, 5, 5)))
@patch("src.numerical_methods.explicit_solver.compute_pressure_divergence", return_value=np.zeros((5, 5, 5)))
@patch("src.numerical_methods.explicit_solver.solve_poisson_for_phi", return_value=np.zeros((5, 5, 5)))
@patch("src.numerical_methods.explicit_solver.apply_pressure_correction", side_effect=lambda u, p, phi, m, dt, rho: (u * 0.5, p + 0.1))
@patch("src.numerical_methods.explicit_solver.apply_boundary_conditions", side_effect=lambda u, p, f, m, is_tentative_step: (u, p))
def test_explicit_solver_step_mocked(
    mock_boundary,
    mock_pcorr,
    mock_poisson,
    mock_divergence,
    mock_diffusion,
    mock_advection,
    initial_velocity,
    mesh_and_props
):
    pressure = np.zeros((5, 5, 5))
    mesh_info, fluid_props = mesh_and_props
    solver = ExplicitSolver(fluid_props, mesh_info, dt=0.1)

    velocity_out, pressure_out = solver.step(initial_velocity, pressure)

    assert velocity_out.shape == initial_velocity.shape
    assert pressure_out.shape == pressure.shape
    assert np.allclose(pressure_out, 0.1), "Pressure field should be offset by +0.1"

    if not np.allclose(initial_velocity, 0):
        assert not np.allclose(velocity_out, initial_velocity), "Velocity should be altered by solver in non-zero case"

    # Ensure mocks were called
    mock_advection.assert_called()
    mock_diffusion.assert_called()
    mock_poisson.assert_called_once()
    mock_pcorr.assert_called_once()
    assert mock_boundary.call_count == 2



