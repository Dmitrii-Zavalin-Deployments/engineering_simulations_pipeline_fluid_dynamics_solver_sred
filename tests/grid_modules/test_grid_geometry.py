# tests/grid_modules/test_grid_geometry.py
# 🧪 Unit tests for src/grid_modules/grid_geometry.py

import pytest
from src.grid_modules.grid_geometry import generate_coordinates

def test_generate_coordinates_valid_domain_1x1x1():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 1,
        "min_y": 0.0, "max_y": 2.0, "ny": 1,
        "min_z": 0.0, "max_z": 3.0, "nz": 1
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 1
    ix, iy, iz, x, y, z = coords[0]
    assert (ix, iy, iz) == (0, 0, 0)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(1.0)
    assert z == pytest.approx(1.5)

def test_generate_coordinates_valid_domain_2x1x1():
    domain = {
        "min_x": 0.0, "max_x": 2.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 2
    assert coords[0][0:3] == (0, 0, 0)
    assert coords[1][0:3] == (1, 0, 0)
    assert coords[0][3] == pytest.approx(0.5)
    assert coords[1][3] == pytest.approx(1.5)

def test_generate_coordinates_missing_keys():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 1,
        "min_y": 0.0, "max_y": 1.0  # missing ny, z keys
    }
    with pytest.raises(ValueError) as excinfo:
        generate_coordinates(domain)
    assert "Missing domain keys" in str(excinfo.value)

def test_generate_coordinates_non_numeric_values():
    domain = {
        "min_x": "a", "max_x": "b", "nx": "c",
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    with pytest.raises(ValueError) as excinfo:
        generate_coordinates(domain)
    assert "Domain must contain numeric bounds" in str(excinfo.value)

def test_generate_coordinates_zero_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 0,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 1
    }
    with pytest.raises(ValueError) as excinfo:
        generate_coordinates(domain)
    assert "Resolution values must be greater than zero" in str(excinfo.value)



