# tests/test_solver_core/test_advection_term.py

import numpy as np
import pytest
from src.numerical_methods.advection import compute_advection_term

def test_zero_velocity_scalar_field_gives_zero_advection():
    shape = (4, 4, 4)
    scalar_field = np.random.rand(*shape)
    velocity = np.zeros(shape + (3,))
    mesh = {"grid_shape": shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    result = compute_advection_term(scalar_field, velocity, mesh)
    assert np.allclose(result, 0.0), "Scalar field should not advect if velocity is zero"

def test_uniform_scalar_field_with_uniform_flow_remains_constant():
    shape = (5, 5, 5)
    field = np.ones(shape) * 3.0
    velocity = np.ones(shape + (3,))
    mesh = {"grid_shape": shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    adv = compute_advection_term(field, velocity, mesh)
    assert np.allclose(adv, 0.0), "Uniform scalar field should not change under uniform flow"

def test_step_scalar_field_advects_properly():
    shape = (5, 5, 5)
    u_field = np.zeros(shape)
    u_field[2:, :, :] = 1.0  # Step in x-direction

    velocity = np.zeros(shape + (3,))
    velocity[..., 0] = 1.0  # Positive x-velocity

    mesh = {"grid_shape": shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(u_field, velocity, mesh)

    assert adv[2, 2, 2] < 0.0, "Should have outgoing flux at step edge"

def test_uniform_vector_field_yields_zero_advection():
    shape = (6, 6, 6)
    vector_field = np.ones(shape + (3,)) * 2.5
    velocity = np.ones(shape + (3,)) * 1.0
    mesh = {"grid_shape": shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    adv = compute_advection_term(vector_field, velocity, mesh)
    assert np.allclose(adv, 0.0, atol=1e-10), "Vector field should not advect if uniform"

def test_vector_field_gradient_advects_components_independently():
    shape = (5, 5, 5)
    vector_field = np.zeros(shape + (3,))
    vector_field[..., 0] = np.linspace(0, 1, shape[0])[:, None, None]  # x-component gradient

    velocity = np.zeros_like(vector_field)
    velocity[..., 0] = 1.0  # Flow in x-direction only

    mesh = {"grid_shape": shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(vector_field, velocity, mesh)

    # Only x-component should have nonzero advection
    assert not np.allclose(adv[..., 0], 0), "X-component should be advected"
    assert np.allclose(adv[..., 1:], 0), "Y and Z components should remain unchanged"



