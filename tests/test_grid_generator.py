# tests/test_grid_generator.py
# 🧪 Validates grid generation with and without masking logic, including shape, encoding, and error cases

import pytest
from src.grid_generator import generate_grid, generate_grid_with_mask
from src.grid_modules.cell import Cell

def domain():
    return {
        "min_x": 0.0, "max_x": 2.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }

def initial_conditions():
    return {
        "velocity": [1.0, 0.0, 0.0],
        "pressure": 5.0
    }

def geometry_mask():
    return {
        "geometry_mask_shape": [2, 1, 1],
        "geometry_mask_flat": [1, 0],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }

def test_generate_grid_basic_success():
    result = generate_grid(domain(), initial_conditions())
    assert isinstance(result, list)
    assert all(isinstance(c, Cell) for c in result)
    assert len(result) == 2  # nx * ny * nz = 2
    assert all(c.velocity == [1.0, 0.0, 0.0] for c in result)

def test_generate_grid_missing_domain_key_raises():
    bad = {**domain()}
    bad.pop("max_x")
    with pytest.raises(ValueError, match="Missing domain keys"):
        generate_grid(bad, initial_conditions())

def test_generate_grid_with_mask_correctly_sets_fluid_mask():
    result = generate_grid_with_mask(domain(), initial_conditions(), geometry_mask())
    assert len(result) == 2
    assert result[0].fluid_mask is True
    assert result[1].fluid_mask is False
    assert result[0].velocity == [1.0, 0.0, 0.0]
    assert result[1].velocity is None

def test_generate_grid_with_mask_shape_mismatch_raises():
    geometry = {**geometry_mask(), "geometry_mask_shape": [3, 1, 1]}
    with pytest.raises(ValueError, match="does not match domain resolution"):
        generate_grid_with_mask(domain(), initial_conditions(), geometry)

def test_generate_grid_with_mask_invalid_flat_mask_length_raises():
    geometry = {**geometry_mask(), "geometry_mask_flat": [1, 0, 1]}
    with pytest.raises(ValueError, match="does not match coordinate count"):
        generate_grid_with_mask(domain(), initial_conditions(), geometry)

def test_generate_grid_with_mask_custom_encoding():
    geom = {
        "geometry_mask_shape": [2, 1, 1],
        "geometry_mask_flat": [9, 2],
        "mask_encoding": {"fluid": 9, "solid": 2},
        "flattening_order": "x-major"
    }
    result = generate_grid_with_mask(domain(), initial_conditions(), geom)
    assert result[0].fluid_mask is True
    assert result[1].fluid_mask is False

def test_generate_grid_with_mask_invalid_order_raises():
    geom = {**geometry_mask(), "flattening_order": "invalid"}
    with pytest.raises(ValueError, match="Failed to decode fluid mask"):
        generate_grid_with_mask(domain(), initial_conditions(), geom)

def test_generate_grid_zero_resolution_warning(caplog):
    d = {**domain(), "nx": 0, "ny": 1, "nz": 1}
    generate_grid(d, initial_conditions())
    assert "Empty grid generated" in caplog.text

def test_generate_grid_with_mask_returns_cell_instances():
    result = generate_grid_with_mask(domain(), initial_conditions(), geometry_mask())
    assert all(isinstance(c, Cell) for c in result)