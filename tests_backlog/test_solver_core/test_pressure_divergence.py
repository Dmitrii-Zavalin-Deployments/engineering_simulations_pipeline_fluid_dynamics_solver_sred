# tests/test_solver_core/test_pressure_divergence.py

import numpy as np
import pytest
from src.numerical_methods.pressure_divergence import (
    compute_pressure_divergence as compute_divergence,
    compute_pressure_gradient,
)

def create_velocity_field(shape, component=0, value=1.0):
    u = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
    u[..., component] = value
    return u

def create_zero_velocity_field(shape):
    return np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))

def create_linearly_increasing_velocity(shape, axis=0):
    padded = (shape[0] + 2, shape[1] + 2, shape[2] + 2)
    velocity = np.zeros(padded + (3,), dtype=np.float64)
    grid = np.linspace(0, 1, padded[axis])
    if axis == 0:
        velocity[..., 0] = grid[:, None, None]
    elif axis == 1:
        velocity[..., 1] = grid[None, :, None]
    elif axis == 2:
        velocity[..., 2] = grid[None, None, :]
    return velocity

def create_quadratic_pressure_field(shape, axis=0):
    padded = (shape[0] + 2, shape[1] + 2, shape[2] + 2)
    p = np.zeros(padded, dtype=np.float64)
    coords = np.linspace(-1, 1, padded[axis]) ** 2
    if axis == 0:
        p[:, :, :] = coords[:, None, None]
    elif axis == 1:
        p[:, :, :] = coords[None, :, None]
    elif axis == 2:
        p[:, :, :] = coords[None, None, :]
    return p

def create_pressure_field(shape, ramp_axis=0):
    padded = (shape[0] + 2, shape[1] + 2, shape[2] + 2)
    p = np.zeros(padded, dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, padded[ramp_axis])
    if ramp_axis == 0:
        p[:, :, :] = ramp[:, None, None]
    elif ramp_axis == 1:
        p[:, :, :] = ramp[None, :, None]
    elif ramp_axis == 2:
        p[:, :, :] = ramp[None, None, :]
    return p

def mesh_metadata(shape, dx=1.0, dy=1.0, dz=1.0):
    return {
        "grid_shape": (shape[0] + 2, shape[1] + 2, shape[2] + 2),
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }

def test_divergence_of_zero_velocity_is_zero():
    shape = (6, 6, 6)
    velocity = create_zero_velocity_field(shape)
    mesh = mesh_metadata(shape)
    div = compute_divergence(velocity, mesh)
    assert np.allclose(div[2:-2, 2:-2, 2:-2], 0.0, atol=1e-12)

def test_pressure_gradient_is_nonzero_for_ramp():
    shape = (6, 6, 6)
    pressure = create_pressure_field(shape, ramp_axis=0)
    mesh = mesh_metadata(shape)
    grad = compute_pressure_gradient(pressure, mesh)
    core = grad[2:-2, 2:-2, 2:-2]
    assert not np.allclose(core[..., 0], 0.0, atol=1e-10)
    assert np.allclose(core[..., 1:], 0.0, atol=1e-12)

def test_divergence_of_pressure_gradient_mimics_laplacian():
    shape = (6, 6, 6)
    pressure = create_quadratic_pressure_field(shape, axis=0)
    mesh = mesh_metadata(shape)
    grad = compute_pressure_gradient(pressure, mesh)
    div = compute_divergence(grad, mesh)

    # Use safer slicing to avoid shrinking past zero with double-stencilpasses
    interior = div[1:-1, 1:-1, 1:-1]

    assert interior.size > 0, "Divergence field has no valid interior"
    mean_val = np.mean(interior)
    assert np.allclose(interior, mean_val, atol=1e-10)

def test_divergence_sign_detects_expansion_or_compression():
    shape = (5, 5, 5)
    velocity = create_linearly_increasing_velocity(shape, axis=0)
    mesh = mesh_metadata(shape)
    div = compute_divergence(velocity, mesh)
    core = div[2:-2, 2:-2, 2:-2]
    assert np.all(core > 0.0)

def test_divergence_of_rotational_field_is_zero():
    shape = (6, 6, 6)
    u = np.zeros((shape[0] + 2, shape[1] + 2, shape[2] + 2, 3))
    X, Y, Z = np.meshgrid(
        np.linspace(-1, 1, shape[0] + 2),
        np.linspace(-1, 1, shape[1] + 2),
        np.linspace(-1, 1, shape[2] + 2),
        indexing="ij"
    )
    u[..., 0] = -Y
    u[..., 1] = X
    mesh = mesh_metadata(shape)
    div = compute_divergence(u, mesh)
    assert np.allclose(div[2:-2, 2:-2, 2:-2], 0.0, atol=1e-10)

def test_anisotropic_spacing_affects_divergence_scale():
    shape = (6, 6, 6)
    velocity = create_linearly_increasing_velocity(shape, axis=0)
    mesh = mesh_metadata(shape, dx=2.0, dy=1.0, dz=0.5)

    nx = mesh["grid_shape"][0] - 2
    spacing = mesh["dx"]
    effective_intervals = nx + 1
    expected = (1.0 / effective_intervals) / spacing

    div = compute_divergence(velocity, mesh)
    core = div[2:-2, 2:-2, 2:-2]
    assert np.allclose(core, expected, atol=1e-10)

def test_divergence_of_velocity_with_boundary_gradient_only():
    shape = (6, 6, 6)
    velocity = create_zero_velocity_field(shape)
    velocity[3, 1:-1, 1:-1, 0] = 1.0
    mesh = mesh_metadata(shape)
    div = compute_divergence(velocity, mesh)
    assert np.any(np.abs(div) > 0.0)



