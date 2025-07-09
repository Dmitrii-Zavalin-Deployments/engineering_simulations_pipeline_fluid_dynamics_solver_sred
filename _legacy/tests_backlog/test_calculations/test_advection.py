import numpy as np
import pytest
from src.numerical_methods.advection import compute_advection_term, advect_velocity

@pytest.fixture
def default_mesh():
    return {"grid_shape": (5, 5, 5), "dx": 1.0, "dy": 1.0, "dz": 1.0}

def test_scalar_field_constant_flow_zero_advection(default_mesh):
    u_field = np.ones((5, 5, 5))
    velocity = np.zeros((5, 5, 5, 3))
    result = compute_advection_term(u_field, velocity, default_mesh)
    assert np.allclose(result, 0.0), "Advection of constant scalar field with no flow should be zero"

def test_vector_field_uniform_flow_zero_advection(default_mesh):
    u_field = np.ones((5, 5, 5, 3))
    velocity = np.zeros((5, 5, 5, 3))
    result = compute_advection_term(u_field, velocity, default_mesh)
    assert np.allclose(result, 0.0), "Advection of constant vector field with no flow should be zero"

def test_scalar_gradient_flow_directional(default_mesh):
    u_field = np.tile(np.linspace(0, 1, 5)[..., None, None], (1, 5, 5))  # Linear in x
    velocity = np.zeros((5, 5, 5, 3))
    velocity[..., 0] = 1.0  # Flow in +x

    result = compute_advection_term(u_field, velocity, default_mesh)
    expected = np.zeros_like(u_field)
    expected[1:-1, 1:-1, 1:-1] = -0.25  # corrected expectation based on discrete spacing
    assert np.allclose(result[1:-1, 1:-1, 1:-1], expected[1:-1, 1:-1, 1:-1]), "Mismatch in advection of linear scalar field"

def test_vector_field_with_flow_componentwise(default_mesh):
    u_field = np.zeros((5, 5, 5, 3))
    x = np.linspace(0, 1, 5)
    for comp in range(3):
        if comp == 0:
            u_field[..., comp] = x[:, None, None]
        elif comp == 1:
            u_field[..., comp] = x[None, :, None]
        else:
            u_field[..., comp] = x[None, None, :]

    velocity = np.ones((5, 5, 5, 3)) * [1.0, 1.0, 1.0]
    result = compute_advection_term(u_field, velocity, default_mesh)
    assert not np.allclose(result, 0.0), "Advection of linear vector field with flow should not be zero"
    assert result.shape == u_field.shape

def test_advect_velocity_returns_expected_shape():
    u = np.ones((7, 7, 7))
    v = np.ones((7, 7, 7)) * 2
    w = np.ones((7, 7, 7)) * 3
    dx = dy = dz = 1.0
    dt = 0.01

    u_star, v_star, w_star = advect_velocity(u, v, w, dx, dy, dz, dt)
    assert u_star.shape == (5, 5, 5)
    assert v_star.shape == (5, 5, 5)
    assert w_star.shape == (5, 5, 5)

def test_advect_velocity_zero_velocity_fields():
    shape = (7, 7, 7)
    zero_u = np.zeros(shape)
    zero_v = np.zeros(shape)
    zero_w = np.zeros(shape)
    dx = dy = dz = 1.0
    dt = 0.01

    u_star, v_star, w_star = advect_velocity(zero_u, zero_v, zero_w, dx, dy, dz, dt)
    assert np.allclose(u_star, 0), "u* should remain zero for zero initial velocity"
    assert np.allclose(v_star, 0), "v* should remain zero for zero initial velocity"
    assert np.allclose(w_star, 0), "w* should remain zero for zero initial velocity"



