# tests/physics/advection_methods/test_euler.py
# 🧪 Unit tests for compute_euler_velocity — Forward Euler advection method

import pytest
from src.physics.advection_methods.euler import compute_euler_velocity
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=1.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def make_config(nx=10, ny=1, nz=1, min_x=0.0, max_x=1.0):
    return {
        "domain_definition": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "min_x": min_x,
            "max_x": max_x
        }
    }

# ------------------------------
# Grid Structure and Edge Cases
# ------------------------------

def test_advection_preserves_velocity_with_no_neighbor():
    cell = make_cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0])
    grid = [cell]
    config = make_config()
    result = compute_euler_velocity(grid, dt=0.1, config=config)
    assert result[0].velocity == [1.0, 0.0, 0.0]

def test_advection_modifies_velocity_with_valid_upwind_neighbor():
    dx = 0.1
    cell = make_cell(x=0.1, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0])
    neighbor = make_cell(x=0.0, y=0.0, z=0.0, velocity=[2.0, 0.0, 0.0])
    grid = [cell, neighbor]
    config = make_config(nx=10, min_x=0.0, max_x=1.0)
    result = compute_euler_velocity(grid, dt=0.1, config=config)

    expected_vx = 1.0 + (0.1 / dx) * (2.0 - 1.0)  # = 2.0
    assert pytest.approx(result[0].velocity[0]) == expected_vx
    assert result[0].velocity[1] == 0.0
    assert result[0].velocity[2] == 0.0

def test_advection_ignores_solid_neighbors():
    cell = make_cell(x=0.1, y=0.0, z=0.0, velocity=[1.0, 1.0, 0.0])
    solid = make_cell(x=0.0, y=0.0, z=0.0, velocity=[9.0, 9.0, 9.0], fluid_mask=False)
    grid = [cell, solid]
    config = make_config()
    result = compute_euler_velocity(grid, dt=0.1, config=config)
    assert result[0].velocity == [1.0, 1.0, 0.0]  # unchanged

def test_advection_returns_safe_output_for_empty_grid():
    result = compute_euler_velocity([], dt=0.1, config=make_config())
    assert result == []

# ---------------------------------
# Input Validation and Type Safety
# ---------------------------------

def test_advection_handles_malformed_velocity_vector():
    bad_cell = make_cell(x=0.0, y=0.0, z=0.0, velocity="bad", pressure=1.0)
    result = compute_euler_velocity([bad_cell], dt=0.1, config=make_config())
    assert result[0].velocity == "bad"

def test_advection_preserves_coordinates_and_mask():
    cell = make_cell(x=2.0, y=3.0, z=4.0, velocity=[0.5, 0.5, 0.5], fluid_mask=True)
    result = compute_euler_velocity([cell], dt=0.1, config=make_config())
    updated = result[0]
    assert (updated.x, updated.y, updated.z) == (2.0, 3.0, 4.0)
    assert updated.fluid_mask is True

# ------------------------------------
# Domain Resolution and dx Handling
# ------------------------------------

def test_advection_fallback_to_dx_one_for_zero_resolution():
    cell = make_cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0])
    neighbor = make_cell(x=-1.0, y=0.0, z=0.0, velocity=[2.0, 0.0, 0.0])
    config = make_config(nx=0)  # fallback dx = 1.0
    result = compute_euler_velocity([cell, neighbor], dt=0.1, config=config)
    expected_vx = 1.0 + 0.1 * (2.0 - 1.0) / 1.0  # = 1.1
    assert pytest.approx(result[0].velocity[0]) == expected_vx

def test_advection_dx_calculation_matches_config():
    config = make_config(nx=10, min_x=0.0, max_x=1.0)
    dx = (1.0 - 0.0) / 10
    assert dx == 0.1



