import numpy as np
import pytest

from src.numerical_methods.implicit_solver import ImplicitSolver
from src.numerical_methods.pressure_divergence import compute_pressure_divergence
from tests.utils.test_utils_velocity import generate_velocity_with_divergence

@pytest.fixture
def small_grid_data():
    shape = (4, 4, 4)
    padded_shape = tuple(s + 2 for s in shape)  # ghost cell padding
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
        "boundary_conditions": {}
    }

    fluid_props = {"density": density, "viscosity": viscosity}
    return velocity_field, pressure_field, mesh_info, fluid_props, dt

def test_implicit_solver_instantiates(small_grid_data):
    _, _, mesh_info, fluid_props, dt = small_grid_data
    solver = ImplicitSolver(fluid_props, mesh_info, dt)
    assert solver.dt == dt
    assert solver.viscosity > 0
    assert isinstance(solver.mesh_info, dict)

def test_step_produces_output_of_same_shape(small_grid_data):
    u, p, mesh_info, fluid_props, dt = small_grid_data
    solver = ImplicitSolver(fluid_props, mesh_info, dt)
    updated_u, updated_p = solver.step(u, p)

    assert updated_u.shape == u.shape
    assert updated_p.shape == p.shape
    assert np.isfinite(updated_u).all()
    assert np.isfinite(updated_p).all()

def test_zero_velocity_field_is_stable(small_grid_data):
    u, p, mesh_info, fluid_props, dt = small_grid_data
    solver = ImplicitSolver(fluid_props, mesh_info, dt)
    new_u, new_p = solver.step(u, p)

    assert np.allclose(new_u, 0.0, atol=1e-10)

def test_divergence_is_reduced_for_nonzero_field(small_grid_data):
    _, p, mesh_info, fluid_props, dt = small_grid_data
    shape = tuple(s + 2 for s in mesh_info["grid_shape"])
    u = generate_velocity_with_divergence(shape, pattern="x-ramp", scale=0.2)

    solver = ImplicitSolver(fluid_props, mesh_info, dt)
    div_before = compute_pressure_divergence(u, mesh_info)
    mean_div_before = np.mean(np.abs(div_before))
    assert mean_div_before > 1e-5, "Precondition failed: divergence too small"

    updated_u, _ = solver.step(u, p)
    div_after = compute_pressure_divergence(updated_u, mesh_info)
    mean_div_after = np.mean(np.abs(div_after))

    assert mean_div_after < mean_div_before, "Solver failed to reduce divergence"

def test_random_velocity_field_survives_step(small_grid_data):
    _, p, mesh_info, fluid_props, dt = small_grid_data
    shape = tuple(s + 2 for s in mesh_info["grid_shape"])
    u = generate_velocity_with_divergence(shape, pattern="random", scale=0.05, seed=42)

    solver = ImplicitSolver(fluid_props, mesh_info, dt)
    new_u, new_p = solver.step(u, p)

    assert np.isfinite(new_u).all()
    assert np.isfinite(new_p).all()
    assert new_u.shape == u.shape



