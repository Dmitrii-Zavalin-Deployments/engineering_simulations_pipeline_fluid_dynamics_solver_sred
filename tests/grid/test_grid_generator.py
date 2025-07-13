# tests/grid/test_grid_generator.py

import pytest
from src.grid_generator import generate_grid_with_mask
from src.grid_modules.cell import Cell

def test_masked_grid_structure_correctness():
    domain = {
        "nx": 3, "ny": 2, "nz": 1,
        "min_x": 0.0, "max_x": 3.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    initial_conditions = {
        "initial_velocity": [1.0, 0.0, 0.0],
        "initial_pressure": 100.0
    }
    geometry = {
        "geometry_mask_flat": [1, 1, 0, 0, 1, 1],
        "geometry_mask_shape": [3, 2, 1],
        "mask_encoding": { "fluid": 1, "solid": 0 },
        "flattening_order": "x-major"
    }

    grid = generate_grid_with_mask(domain, initial_conditions, geometry)
    assert len(grid) == 6
    fluid_count = 0
    solid_count = 0

    for cell in grid:
        assert isinstance(cell, Cell)
        assert isinstance(cell.x, (int, float))
        assert isinstance(cell.y, (int, float))
        assert isinstance(cell.z, (int, float))
        assert isinstance(cell.fluid_mask, bool)
        if cell.fluid_mask:
            fluid_count += 1
            assert isinstance(cell.velocity, list) and len(cell.velocity) == 3
            assert isinstance(cell.pressure, float)
        else:
            solid_count += 1
            assert cell.velocity is None
            assert cell.pressure is None

    assert fluid_count == 4
    assert solid_count == 2

def test_mask_shape_mismatch_raises():
    domain = {
        "nx": 2, "ny": 2, "nz": 1,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    initial_conditions = {
        "initial_velocity": [0.0, 0.0, 0.0],
        "initial_pressure": 0.0
    }
    geometry = {
        "geometry_mask_flat": [1, 1, 0, 0, 1, 1],
        "geometry_mask_shape": [3, 2, 1],
        "mask_encoding": { "fluid": 1, "solid": 0 },
        "flattening_order": "x-major"
    }

    with pytest.raises(ValueError, match="does not match domain resolution"):
        generate_grid_with_mask(domain, initial_conditions, geometry)

def test_masked_cells_sanitization_and_assignment():
    domain = {
        "nx": 2, "ny": 1, "nz": 1,
        "min_x": 0.0, "max_x": 2.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }
    initial_conditions = {
        "initial_velocity": [0.5, 0.5, 0.0],
        "initial_pressure": 1.5
    }
    geometry = {
        "geometry_mask_flat": [1, 0],
        "geometry_mask_shape": [2, 1, 1],
        "mask_encoding": { "fluid": 1, "solid": 0 },
        "flattening_order": "x-major"
    }

    grid = generate_grid_with_mask(domain, initial_conditions, geometry)
    assert len(grid) == 2
    assert grid[0].fluid_mask is True
    assert grid[0].velocity == [0.5, 0.5, 0.0]
    assert grid[0].pressure == 1.5

    assert grid[1].fluid_mask is False
    assert grid[1].velocity is None
    assert grid[1].pressure is None



