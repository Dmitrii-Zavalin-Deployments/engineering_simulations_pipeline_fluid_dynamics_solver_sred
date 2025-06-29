import numpy as np
import pytest
from src.numerical_methods.pressure_correction import calculate_gradient, apply_pressure_correction

@pytest.fixture
def sample_fields():
    # Define a small cubic grid
    nx, ny, nz = 5, 5, 5
    phi = np.ones((nx, ny, nz)) * 2.0
    p_field = np.zeros((nx+2, ny+2, nz+2))
    velocity = np.zeros((nx+2, ny+2, nz+2, 3))
    mesh_info = {"dx": 1.0, "dy": 1.0, "dz": 1.0}
    return velocity, p_field, phi, mesh_info

def test_calculate_gradient_axis0_returns_expected_shape():
    field = np.random.rand(5, 5, 5)
    grad = calculate_gradient(field, h=1.0, axis=0)
    assert grad.shape == field.shape
    assert np.isfinite(grad).all()

def test_calculate_gradient_invalid_axis_raises():
    field = np.zeros((5, 5, 5))
    with pytest.raises(ValueError, match="Axis must be 0, 1, or 2"):
        calculate_gradient(field, h=1.0, axis=3)

def test_apply_pressure_correction_updates_pressure(sample_fields):
    velocity, p_field, phi, mesh_info = sample_fields
    dt, rho = 0.1, 1.0
    corrected_velocity, updated_pressure = apply_pressure_correction(
        velocity, p_field, phi, mesh_info, time_step=dt, density=rho
    )
    assert corrected_velocity.shape == velocity.shape
    assert updated_pressure.shape == p_field.shape

    # Check that pressure was updated in the interior
    assert np.allclose(updated_pressure[1:-1, 1:-1, 1:-1], phi)
    assert not np.allclose(corrected_velocity[1:-1, 1:-1, 1:-1, 0], 0.0)

def test_correction_preserves_ghost_boundaries(sample_fields):
    velocity, p_field, phi, mesh_info = sample_fields
    corrected_velocity, updated_pressure = apply_pressure_correction(
        velocity, p_field, phi, mesh_info, time_step=0.1, density=1.0
    )

    # Ensure ghost layers remain unchanged
    assert np.allclose(corrected_velocity[0, :, :, :], 0.0)
    assert np.allclose(updated_pressure[0, :, :], 0.0)



