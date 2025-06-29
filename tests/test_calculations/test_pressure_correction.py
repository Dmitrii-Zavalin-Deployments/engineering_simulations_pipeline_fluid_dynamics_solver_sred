import numpy as np
import pytest
from src.numerical_methods.pressure_correction import calculate_gradient, apply_pressure_correction

@pytest.fixture
def sample_fields_varying_phi():
    # Define a 3D parabolic φ field for meaningful gradient
    shape = (5, 5, 5)
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    phi = X**2 + Y**2 + Z**2  # Non-constant φ with spatial gradient

    p_field = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2))
    velocity = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
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

def test_apply_pressure_correction_with_varying_phi(sample_fields_varying_phi):
    velocity, p_field, phi, mesh_info = sample_fields_varying_phi
    dt, rho = 0.1, 1.0
    corrected_velocity, updated_pressure = apply_pressure_correction(
        velocity, p_field, phi, mesh_info, time_step=dt, density=rho
    )
    assert corrected_velocity.shape == velocity.shape
    assert updated_pressure.shape == p_field.shape

    # Pressure should be updated to φ in the interior
    assert np.allclose(updated_pressure[1:-1, 1:-1, 1:-1], phi, atol=1e-12)

    # Velocity should now have non-zero values due to ∇φ ≠ 0
    magnitude = np.linalg.norm(corrected_velocity[1:-1, 1:-1, 1:-1, :])
    assert magnitude > 0.0, "Velocity field should be affected by pressure gradient"

def test_pressure_correction_preserves_ghost_cells(sample_fields_varying_phi):
    velocity, p_field, phi, mesh_info = sample_fields_varying_phi
    corrected_velocity, updated_pressure = apply_pressure_correction(
        velocity, p_field, phi, mesh_info, time_step=0.1, density=1.0
    )
    # Check ghost layers remain untouched
    assert np.allclose(corrected_velocity[0, :, :, :], 0.0)
    assert np.allclose(updated_pressure[0, :, :], 0.0)



