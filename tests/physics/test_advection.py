# tests/physics/test_advection.py
# 🧪 Integration tests for advection.py — default Euler method routing

import pytest
from src.physics.advection import compute_advection
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, pressure=1.0, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

# ------------------------------
# compute_advection() test cases
# ------------------------------

def test_advection_preserves_velocity_for_simple_grid():
    grid = [make_cell(0, 0, 0, [1.0, 0.0, 0.0])]
    config = {"simulation_parameters": {"time_step": 0.1}}
    dt = 0.1
    updated = compute_advection(grid, dt, config)

    assert isinstance(updated, list)
    assert len(updated) == 1
    assert updated[0].velocity == [1.0, 0.0, 0.0]

def test_advection_does_not_modify_solid_cells():
    solid_cell = make_cell(1, 0, 0, None, pressure=None, fluid_mask=False)
    grid = [solid_cell]
    config = {"simulation_parameters": {"time_step": 0.05}}
    dt = 0.05
    result = compute_advection(grid, dt, config)

    assert len(result) == 1
    assert result[0].fluid_mask is False
    assert result[0].velocity is None
    assert result[0].pressure is None

def test_advection_handles_mixed_grid():
    grid = [
        make_cell(0, 0, 0, [1.0, 1.0, 0.0], fluid_mask=True),
        make_cell(1, 0, 0, None, pressure=None, fluid_mask=False),
        make_cell(2, 0, 0, [0.0, 0.0, 1.0], fluid_mask=True)
    ]
    config = {"simulation_parameters": {"time_step": 0.1}}
    dt = 0.1
    result = compute_advection(grid, dt, config)

    assert len(result) == 3
    assert result[0].velocity == [1.0, 1.0, 0.0]
    assert result[1].velocity is None
    assert result[2].velocity == [0.0, 0.0, 1.0]

def test_advection_returns_safe_output_for_empty_grid():
    updated = compute_advection([], dt=0.1, config={})
    assert updated == []

def test_advection_preserves_cell_coordinates_and_mask():
    grid = [make_cell(3, 2, 1, [0.5, 0.5, 0.5])]
    config = {"simulation_parameters": {"time_step": 0.1}}
    result = compute_advection(grid, dt=0.1, config=config)

    cell = result[0]
    assert (cell.x, cell.y, cell.z) == (3, 2, 1)
    assert cell.fluid_mask is True
    assert isinstance(cell.velocity, list)
    assert len(cell.velocity) == 3

def test_advection_ignores_malformed_velocity_vector():
    bad_cell = make_cell(0, 0, 0, "invalid_velocity")
    result = compute_advection([bad_cell], dt=0.1, config={})

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].velocity == "invalid_velocity"  # preserved as-is



