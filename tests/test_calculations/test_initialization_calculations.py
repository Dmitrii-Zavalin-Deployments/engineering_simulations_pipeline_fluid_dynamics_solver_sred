import numpy as np
import pytest
from src.physics.initialization import initialize_fields

def test_initialize_fields_uniform_values():
    num_nodes = 10
    initial_velocity = [1.0, 0.0, -0.5]
    initial_pressure = 3.14

    velocity, pressure = initialize_fields(num_nodes, initial_velocity, initial_pressure)

    assert velocity.shape == (10, 3)
    assert pressure.shape == (10,)
    assert np.allclose(velocity, initial_velocity)
    assert np.allclose(pressure, initial_pressure)

def test_initialize_fields_zero_values():
    num_nodes = 5
    velocity, pressure = initialize_fields(num_nodes, [0.0, 0.0, 0.0], 0.0)

    assert np.all(velocity == 0.0)
    assert np.all(pressure == 0.0)

def test_initialize_fields_types_are_correct():
    velocity, pressure = initialize_fields(3, [1.0, 2.0, 3.0], 2.5)

    assert isinstance(velocity, np.ndarray)
    assert isinstance(pressure, np.ndarray)
    assert velocity.dtype == np.float64 or velocity.dtype == np.float32
    assert pressure.dtype == np.float64 or pressure.dtype == np.float32

def test_initialize_fields_does_not_mutate_input():
    init_velocity = [1.0, -1.0, 0.5]
    init_pressure = 8.8
    _ = initialize_fields(6, init_velocity, init_pressure)

    # Confirm inputs are unaltered
    assert init_velocity == [1.0, -1.0, 0.5]
    assert init_pressure == 8.8



