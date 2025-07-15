# tests/test_grid_geometry.py
# 🧪 Unit tests for generate_coordinates — validates grid indexing and physical coordinate mapping

import pytest
from src.grid_modules.grid_geometry import generate_coordinates

def test_basic_grid_generation():
    domain = {
        "min_x": 0.0, "max_x": 2.0, "nx": 2,
        "min_y": 0.0, "max_y": 2.0, "ny": 2,
        "min_z": 0.0, "max_z": 2.0, "nz": 2
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 8  # 2 x 2 x 2
    assert (0, 0, 0, 0.5, 0.5, 0.5) in coords
    assert (1, 1, 1, 1.5, 1.5, 1.5) in coords

def test_single_cell_centered():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 1,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 1
    ix, iy, iz, x, y, z = coords[0]
    assert (ix, iy, iz) == (0, 0, 0)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)
    assert z == pytest.approx(0.5)

@pytest.mark.parametrize("missing_key", [
    "min_x", "max_x", "nx",
    "min_y", "max_y", "ny",
    "min_z", "max_z", "nz"
])
def test_missing_keys_raises(missing_key):
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 1,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    del domain[missing_key]
    with pytest.raises(ValueError, match="Missing domain keys"):
        generate_coordinates(domain)

def test_non_numeric_values_raises():
    domain = {
        "min_x": "a", "max_x": "b", "nx": "c",
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    with pytest.raises(ValueError, match="numeric bounds"):
        generate_coordinates(domain)

def test_zero_resolution_raises():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 0,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    with pytest.raises(ValueError, match="greater than zero"):
        generate_coordinates(domain)

def test_negative_resolution_raises():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": -1,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    with pytest.raises(ValueError, match="greater than zero"):
        generate_coordinates(domain)

def test_coordinates_are_centered():
    domain = {
        "min_x": -1.0, "max_x": 1.0, "nx": 2,
        "min_y": -2.0, "max_y": 2.0, "ny": 2,
        "min_z": -3.0, "max_z": 3.0, "nz": 2
    }
    coords = generate_coordinates(domain)
    expected_xs = [-0.5, 0.5]
    expected_ys = [-1.0, 1.0]
    expected_zs = [-1.5, 1.5]
    physical = [(x, y, z) for (_, _, _, x, y, z) in coords]
    for x, y, z in physical:
        assert x in expected_xs
        assert y in expected_ys
        assert z in expected_zs