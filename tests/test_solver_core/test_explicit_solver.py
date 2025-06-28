import numpy as np
import pytest

from src.numerical_methods.explicit_solver import ExplicitSolver
from src.numerical_methods.pressure_divergence import compute_pressure_divergence

@pytest.fixture
def small_grid_data():
    shape = (4, 4, 4)
    padded_shape = tuple(s + 2 for s in shape)  # Add ghost cells
    dt = 0.01
    viscosity = 0.01
    density = 1.0

    velocity_field = np.zeros(padded_shape + (3,), dtype=np.float64)
    pressure_field = np.ones(padded_shape, dtype=np.float64) * 101325.0

    mesh_info = {
        "grid_shape": list(shape),
        "dx": 1.0, "dy": 1.0, "dz": 1.0,
        "min_x": 0.0, "max_x": shape[0],
        "min_y": 0.0, "max_y": shape[1],
        "min_z": 0.0, "max_z": shape[2],
        "boundary_conditions": {}  # no-op BCs for test
    }

    fluid_props = {"density": density, "viscosity": viscosity}
    return velocity_field, pressure_field, mesh_info, fluid_props, dt

def test_explicit_solver_instantiates(small_grid_data):
    _, _, mesh_info, fluid_props, dt = small_grid_data
    solver = ExplicitSolver(fluid_props, mesh_info, dt)
    assert solver.density == 1.0
    assert solver.viscosity == 0.01
    assert solver.dt == dt

def test_step_returns_correct_shape(small_grid_data):
    u, p, mesh_info, fluid_props, dt = small_grid_data
    solver = ExplicitSolver(fluid_props, mesh_info, dt)
    updated_u, updated_p = solver.step(u, p)

    assert updated_u.shape == u.shape
    assert updated_p.shape == p.shape
    assert np.isfinite(updated_u).all()
    assert np.isfinite(updated_p).all()

def test_step_preserves_zero_velocity(small_grid_data):
    u, p, mesh_info, fluid_props, dt = small_grid_data
    solver = ExplicitSolver(fluid_props, mesh_info, dt)
    new_u, new_p = solver.step(u, p)

    # With no forces or BCs, velocity should remain zero
    assert np.allclose(new_u, 0.0, atol=1e-10)

def test_step_decreases_divergence(small_grid_data):
    u, p, mesh_info, fluid_props, dt = small_grid_data

    # Seed measurable divergence in x-direction
    for i in range(1, u.shape[0] - 1):
        u[i, :, :, 0] = i * 0.1  # linearly increasing uₓ → ∂u/∂x ≠ 0

    div_before = compute_pressure_divergence(u, mesh_info)
    mean_div_before = np.mean(np.abs(div_before))
    assert mean_div_before > 1e-5, "Test precondition failed: divergence too small"

    solver = ExplicitSolver(fluid_props, mesh_info, dt)
    new_u, _ = solver.step(u, p)

    div_after = compute_pressure_divergence(new_u, mesh_info)
    mean_div_after = np.mean(np.abs(div_after))

    assert mean_div_after < mean_div_before, "Solver did not reduce divergence"



