# tests/test_solver_core/test_diffusion_operator.py

import numpy as np
import pytest
from src.numerical_methods.diffusion import (
    apply_diffusion_step,
    compute_diffusion_term,
)

def create_uniform_field(nx, ny, nz, value=1.0):
    field = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)
    field[1:-1, 1:-1, 1:-1] = value
    return field

def create_spike_field(nx, ny, nz, spike_value=10.0):
    field = np.zeros((nx + 2, ny + 2, nz + 2), dtype=np.float64)
    cx, cy, cz = (nx + 1) // 2, (ny + 1) // 2, (nz + 1) // 2
    field[cx, cy, cz] = spike_value
    return field

def mesh_metadata(nx, ny, nz, dx=1.0, dy=1.0, dz=1.0):
    return {
        "grid_shape": (nx + 2, ny + 2, nz + 2),
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }

def apply_zero_gradient_ghost_padding(field):
    field[0] = field[1]
    field[-1] = field[-2]
    field[:, 0] = field[:, 1]
    field[:, -1] = field[:, -2]
    field[:, :, 0] = field[:, :, 1]
    field[:, :, -1] = field[:, :, -2]
    return field

def test_apply_diffusion_uniform_field_stays_constant():
    nx, ny, nz = 6, 6, 6
    dt = 0.01
    nu = 0.1
    field = create_uniform_field(nx, ny, nz, value=5.0)
    field = apply_zero_gradient_ghost_padding(field)
    mesh_info = mesh_metadata(nx, ny, nz)

    result = apply_diffusion_step(field.copy(), nu, mesh_info, dt)
    core = result[2:-2, 2:-2, 2:-2]
    assert np.allclose(core, 5.0, atol=1e-10)

def test_apply_diffusion_smooths_peak():
    nx, ny, nz = 5, 5, 5
    dt = 0.01
    nu = 0.1
    mesh_info = mesh_metadata(nx, ny, nz)
    u = create_spike_field(nx, ny, nz, spike_value=10.0)
    u = apply_zero_gradient_ghost_padding(u)

    u_new = apply_diffusion_step(u.copy(), nu, mesh_info, dt)
    center = (nx + 1) // 2
    peak_before = u[center, center, center]
    peak_after = u_new[center, center, center]
    neighbors = [
        u_new[center + 1, center, center],
        u_new[center - 1, center, center],
        u_new[center, center + 1, center],
        u_new[center, center - 1, center],
        u_new[center, center, center + 1],
        u_new[center, center, center - 1],
    ]
    assert peak_after < peak_before
    assert all(n > 0.0 for n in neighbors)

def test_compute_diffusion_term_returns_zero_for_uniform_field():
    nx, ny, nz = 4, 4, 4
    nu = 0.1
    mesh_info = mesh_metadata(nx, ny, nz)
    field = create_uniform_field(nx, ny, nz, value=2.5)
    field = apply_zero_gradient_ghost_padding(field)
    diff = compute_diffusion_term(field, nu, mesh_info)
    interior = diff[2:-2, 2:-2, 2:-2]
    assert np.allclose(interior, 0.0, atol=1e-12)

def test_compute_diffusion_term_returns_expected_shape():
    nx, ny, nz = 5, 4, 3
    mesh_info = mesh_metadata(nx, ny, nz)
    field = np.random.rand(nx + 2, ny + 2, nz + 2)
    diff = compute_diffusion_term(field, viscosity=0.2, mesh_info=mesh_info)
    assert diff.shape == field.shape

def test_vector_field_component_diffusion():
    nx, ny, nz = 6, 6, 6
    mesh_info = mesh_metadata(nx, ny, nz)
    field = np.zeros((nx + 2, ny + 2, nz + 2, 3), dtype=np.float64)
    field[3, 3, 3, 0] = 5.0

    field[0] = field[1]
    field[-1] = field[-2]
    field[:, 0] = field[:, 1]
    field[:, -1] = field[:, -2]
    field[:, :, 0] = field[:, :, 1]
    field[:, :, -1] = field[:, :, -2]

    diff = compute_diffusion_term(field, viscosity=0.1, mesh_info=mesh_info)
    assert diff.shape == field.shape
    assert diff[3, 3, 3, 0] < 0.0
    assert np.allclose(diff[..., 1:], 0.0, atol=1e-10)



