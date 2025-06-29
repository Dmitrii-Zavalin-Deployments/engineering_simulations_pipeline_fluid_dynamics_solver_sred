import numpy as np
import pytest
from src.numerical_methods.diffusion import compute_diffusion_term, apply_diffusion_step

@pytest.fixture
def mesh():
    return {"grid_shape": (5, 5, 5), "dx": 1.0, "dy": 1.0, "dz": 1.0}

def test_diffusion_of_constant_field_is_zero(mesh):
    scalar_field = np.ones((5, 5, 5))
    result = compute_diffusion_term(scalar_field, viscosity=0.1, mesh_info=mesh)
    assert np.allclose(result, 0.0), "Laplacian of constant field should be zero"

def test_diffusion_scalar_linear_x(mesh):
    x = np.linspace(0, 1, 5)
    field = np.tile(x[:, None, None], (1, 5, 5))  # linear ramp in x-direction
    result = compute_diffusion_term(field, viscosity=1.0, mesh_info=mesh)
    assert np.allclose(result, 0.0), "Laplacian of linear scalar field should be zero"

def test_diffusion_scalar_quadratic_x():
    x = np.linspace(-1, 1, 5)
    dx = x[1] - x[0]
    field = np.tile((x**2)[..., None, None], (1, 5, 5))
    mesh = {"grid_shape": (5, 5, 5), "dx": dx, "dy": 1.0, "dz": 1.0}
    result = compute_diffusion_term(field, viscosity=1.0, mesh_info=mesh)
    expected = np.zeros_like(field)
    expected[1:-1, 1:-1, 1:-1] = 2.0  # d²(x²)/dx² = 2
    assert np.allclose(result[1:-1, 1:-1, 1:-1], expected[1:-1, 1:-1, 1:-1]), "Incorrect Laplacian of x²"

def test_diffusion_vector_uniform_field_is_zero(mesh):
    vector_field = np.ones((5, 5, 5, 3))
    result = compute_diffusion_term(vector_field, viscosity=0.1, mesh_info=mesh)
    assert np.allclose(result, 0.0), "Vector diffusion of uniform field should be zero"

def test_apply_diffusion_step_advances_correctly(mesh):
    field = np.zeros((5, 5, 5))
    field[2, 2, 2] = 1.0  # Dirac-like impulse
    updated = apply_diffusion_step(field, diffusion_coefficient=1.0, mesh_info=mesh, dt=0.1)
    assert np.sum(updated > 0) > 1, "Impulse should diffuse outward"
    assert updated[2, 2, 2] < 1.0, "Peak value should decrease due to diffusion"



