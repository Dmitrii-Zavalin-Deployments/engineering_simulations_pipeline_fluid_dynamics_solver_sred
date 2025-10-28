import pytest
from src.grid_generator import generate_grid, generate_grid_with_mask
from src.grid_modules.cell import Cell

@pytest.fixture
def domain():
    return {
        "min_x": 0.0, "max_x": 1.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 2,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }

@pytest.fixture
def initial_conditions():
    return {
        "initial_velocity": [1.0, 0.0, 0.0],
        "initial_pressure": 101325.0
    }

@pytest.fixture
def geometry_valid():
    return {
        "geometry_mask_flat": [1, 1, 0, 1],
        "geometry_mask_shape": [2, 2, 1],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }

@pytest.fixture
def geometry_mismatch():
    return {
        "geometry_mask_flat": [1, 1, 0, 1],
        "geometry_mask_shape": [2, 1, 2],  # mismatched shape
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }

def test_generate_grid_returns_cells(domain, initial_conditions):
    cells = generate_grid(domain, initial_conditions)
    assert isinstance(cells, list)
    assert all(isinstance(c, Cell) for c in cells)
    assert all(c.fluid_mask is True for c in cells)
    assert all(c.velocity is not None and c.pressure is not None for c in cells)

def test_generate_grid_with_mask_applies_fluid_mask(domain, initial_conditions, geometry_valid):
    cells = generate_grid_with_mask(domain, initial_conditions, geometry_valid)
    assert len(cells) == 4
    fluid_cells = [c for c in cells if c.fluid_mask]
    solid_cells = [c for c in cells if not c.fluid_mask]
    assert len(fluid_cells) == 3
    assert len(solid_cells) == 1
    for c in fluid_cells:
        assert c.velocity is not None and c.pressure is not None
    for c in solid_cells:
        assert c.velocity is None and c.pressure is None

def test_generate_grid_with_mask_shape_mismatch_raises(domain, initial_conditions, geometry_mismatch):
    with pytest.raises(ValueError) as e:
        generate_grid_with_mask(domain, initial_conditions, geometry_mismatch)
    assert "does not match domain resolution" in str(e.value)

def test_generate_grid_with_mask_invalid_mask_length(domain, initial_conditions, geometry_valid):
    geometry_valid["geometry_mask_flat"] = [1, 0]  # too short
    with pytest.raises(ValueError) as e:
        generate_grid_with_mask(domain, initial_conditions, geometry_valid)
    assert "does not match coordinate count" in str(e.value)

def test_generate_grid_missing_keys_raises(initial_conditions):
    bad_domain = {"min_x": 0.0, "max_x": 1.0, "nx": 2}  # missing keys
    with pytest.raises(ValueError) as e:
        generate_grid(bad_domain, initial_conditions)
    assert "Missing domain keys" in str(e.value)



