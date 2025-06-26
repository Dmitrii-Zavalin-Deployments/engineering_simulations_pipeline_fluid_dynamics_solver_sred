# tests/test_solver_core/test_advection_sanity.py

import numpy as np
import pytest
from src.numerical_methods.advection import compute_advection_term

def create_scalar_field_with_step(shape, step_position, low=0.0, high=1.0):
    """
    Create a padded scalar field where values are low to the left of step_position in x,
    and high to the right.
    """
    padded_shape = (shape[0] + 2, shape[1] + 2, shape[2] + 2)
    field = np.ones(padded_shape) * low
    field[step_position:, 1:-1, 1:-1] = high
    return field

def create_velocity_field(shape, direction='x', magnitude=1.0):
    """
    Create a padded velocity field with uniform flow in a given direction.
    """
    padded_shape = (shape[0] + 2, shape[1] + 2, shape[2] + 2, 3)
    velocity = np.zeros(padded_shape)
    dir_index = {'x': 0, 'y': 1, 'z': 2}[direction]
    velocity[..., dir_index] = magnitude
    return velocity

def test_advection_step_detected():
    shape = (8, 4, 4)
    step_x = 4  # step at x=4
    field = create_scalar_field_with_step(shape, step_x)
    velocity = create_velocity_field(shape, direction='x', magnitude=1.0)

    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}
    adv = compute_advection_term(field, velocity, mesh)

    # Advection should be non-zero near the step
    inspected = adv[step_x - 1 : step_x + 2, 2:3, 2:3]
    assert np.any(np.abs(inspected) > 1e-6), f"Expected non-zero flux at step. Got: {inspected}"

def test_advection_zero_when_uniform_field():
    shape = (6, 6, 6)
    field = np.ones((shape[0] + 2, shape[1] + 2, shape[2] + 2)) * 5.0
    velocity = create_velocity_field(shape, direction='x', magnitude=1.0)
    mesh = {"grid_shape": field.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    adv = compute_advection_term(field, velocity, mesh)
    core = adv[2:-2, 2:-2, 2:-2]
    assert np.allclose(core, 0.0, atol=1e-10), "Uniform field should yield zero advection"

def test_advection_directional_behavior():
    shape = (8, 4, 4)
    field_fwd = create_scalar_field_with_step(shape, step_position=4, low=0.0, high=2.0)
    velocity_fwd = create_velocity_field(shape, direction='x', magnitude=1.0)
    velocity_bwd = create_velocity_field(shape, direction='x', magnitude=-1.0)
    mesh = {"grid_shape": field_fwd.shape, "dx": 1.0, "dy": 1.0, "dz": 1.0}

    adv_fwd = compute_advection_term(field_fwd, velocity_fwd, mesh)
    adv_bwd = compute_advection_term(field_fwd, velocity_bwd, mesh)

    # Inspect flux on either side of step
    assert not np.allclose(adv_fwd, 0.0), "Forward velocity should yield non-zero advection"
    assert not np.allclose(adv_bwd, 0.0), "Backward velocity should also yield non-zero advection"
    assert not np.allclose(adv_fwd, adv_bwd), "Direction should affect advection magnitude/sign"



