import numpy as np
import pytest
from src.numerical_methods.advection import compute_advection_term

def create_padded_field(shape, fill=0.0):
    padded = np.ones((shape[0] + 2, shape[1] + 2, shape[2] + 2)) * fill
    return padded

def create_padded_velocity(shape, fill=0.0):
    return np.ones((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3)) * fill

def test_zero_velocity_scalar_field_gives_zero_advection():
    shape = (4, 4, 4)
    field = create_padded_field(shape, fill=2.0)
    velocity = np.zeros(field.shape + (1,))[:, :, :, 0].copy()
    velocity = np.stack([velocity] * 3, axis=-1)

    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    result = compute_advection_term(field, velocity, mesh)
    assert np.allclose(result[1:-1, 1:-1, 1:-1], 0.0)

def test_uniform_scalar_field_with_uniform_flow_remains_constant():
    shape = (5, 5, 5)
    field = create_padded_field(shape, fill=3.0)
    velocity = create_padded_velocity(shape, fill=1.0)
    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    adv = compute_advection_term(field, velocity, mesh)
    assert np.allclose(adv[1:-1, 1:-1, 1:-1], 0.0, atol=1e-10)

def test_step_scalar_field_advects_properly():
    shape = (5, 5, 5)
    field = create_padded_field(shape)
    field[3:, 1:-1, 1:-1] = 1.0  # sharp step at x=3

    velocity = create_padded_velocity(shape, fill=0.0)
    velocity[..., 0] = 1.0  # x-direction flow

    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)
    assert np.any(adv[2:-2, 2:-2, 2:-2] != 0.0)

def test_uniform_vector_field_yields_zero_advection():
    shape = (6, 6, 6)
    field = np.ones((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3)) * 2.5
    velocity = np.ones_like(field)

    mesh = {"grid_shape": field.shape[:3], "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)
    assert np.allclose(adv[1:-1, 1:-1, 1:-1], 0.0)

def test_vector_field_gradient_advects_components_independently():
    shape = (5, 5, 5)
    field = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
    for i in range(1, shape[0] + 1):
        field[i, 1:-1, 1:-1, 0] = i / shape[0]  # x-gradient

    velocity = np.zeros_like(field)
    velocity[..., 0] = 1.0  # flow in x-direction only

    mesh = {"grid_shape": field.shape[:3], "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)

    core = adv[1:-1, 1:-1, 1:-1]
    assert not np.allclose(core[..., 0], 0), "X-component should be advected"
    assert np.allclose(core[..., 1:], 0), "Y/Z components should remain unchanged"



