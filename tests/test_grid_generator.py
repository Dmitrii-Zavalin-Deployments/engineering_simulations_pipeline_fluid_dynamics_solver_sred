# ✅ Unit Test Suite — Grid Generator
# 📄 Full Path: tests/test_grid_generator.py

import pytest
from src.grid_generator import generate_grid, generate_grid_with_mask
from src.grid_modules.cell import Cell

def test_generate_grid_valid_input():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    initial = {"velocity": [1.0, 0.0, 0.0], "pressure": 0.2}
    grid = generate_grid(domain, initial)
    assert len(grid) == 1
    assert isinstance(grid[0], Cell)
    assert grid[0].fluid_mask is True
    assert grid[0].velocity == [1.0, 0.0, 0.0]
    assert grid[0].pressure == 0.2

def test_generate_grid_missing_keys_raises():
    domain = {"min_x": 0, "max_x": 1, "nx": 1}
    initial = {}
    with pytest.raises(ValueError) as e:
        generate_grid(domain, initial)
    assert "Missing domain keys" in str(e.value)

def test_generate_grid_with_mask_valid():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 2,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    initial = {"velocity": [0.5, 0.0, 0.0], "pressure": 0.1}
    geometry = {
        "geometry_mask_shape": [2, 1, 1],
        "geometry_mask_flat": [1, 0],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }
    grid = generate_grid_with_mask(domain, initial, geometry)
    assert len(grid) == 2
    assert isinstance(grid[0], Cell)
    assert grid[0].fluid_mask is True
    assert grid[1].fluid_mask is False
    assert grid[0].velocity == [0.5, 0.0, 0.0]
    assert grid[1].velocity is None
    assert grid[1].pressure is None

def test_geometry_shape_mismatch_raises():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    geometry = {
        "geometry_mask_shape": [2, 1, 1],
        "geometry_mask_flat": [1, 1]
    }
    initial = {}
    with pytest.raises(ValueError) as e:
        generate_grid_with_mask(domain, initial, geometry)
    assert "does not match domain resolution" in str(e.value)

def test_mask_length_mismatch_raises():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    geometry = {
        "geometry_mask_shape": [1, 1, 1],
        "geometry_mask_flat": [1, 1]  # too long
    }
    initial = {}
    with pytest.raises(ValueError) as e:
        generate_grid_with_mask(domain, initial, geometry)
    assert "does not match coordinate count" in str(e.value)



