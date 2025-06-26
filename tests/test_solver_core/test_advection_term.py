# tests/test_solver_core/test_advection_term.py

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
    velocity = create_padded_velocity(shape, fill=0.0)

    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    result = compute_advection_term(field, velocity, mesh)
    interior = result[2:-2, 2:-2, 2:-2]
    assert np.allclose(interior, 0.0, atol=1e-12)

def test_uniform_scalar_field_with_uniform_flow_remains_constant():
    shape = (5, 5, 5)
    field = create_padded_field(shape, fill=3.0)
    velocity = create_padded_velocity(shape, fill=1.0)
    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    adv = compute_advection_term(field, velocity, mesh)
    interior = adv[2:-2, 2:-2, 2:-2]
    assert np.allclose(interior, 0.0, atol=1e-10)

def test_step_scalar_field_advects_properly():
    shape = (5, 5, 5)
    field = create_padded_field(shape)
    field[2:-1, 2:-2, 2:-2] = 1.0  # sharp step inside domain

    velocity = create_padded_velocity(shape, fill=0.0)
    velocity[..., 0] = 1.0  # x-direction flow

    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)
    assert np.any(np.abs(adv[2:-2, 2:-2, 2:-2]) > 1e-8), "Advection term should be non-zero near step"

def test_uniform_vector_field_yields_zero_advection():
    shape = (6, 6, 6)
    field = np.ones((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3)) * 2.5
    velocity = np.ones_like(field)

    mesh = {"grid_shape": field.shape[:3], "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)
    core = adv[2:-2, 2:-2, 2:-2]
    assert np.allclose(core, 0.0, atol=1e-10), "Uniform vector field should not advect"

def test_vector_field_gradient_advects_components_independently():
    shape = (5, 5, 5)
    field = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
    for i in range(1, shape[0] + 1):
        field[i, 2:-2, 2:-2, 0] = float(i)  # steeper x-gradient

    velocity = np.zeros_like(field)
    velocity[..., 0] = 1.0  # x-direction flow

    mesh = {"grid_shape": field.shape[:3], "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)

    core = adv[2:-2, 2:-2, 2:-2]
    assert not np.allclose(core[..., 0], 0.0, atol=1e-10), "X-component should be advected"
    assert np.allclose(core[..., 1:], 0.0, atol=1e-12), "Y/Z components should remain zero"



