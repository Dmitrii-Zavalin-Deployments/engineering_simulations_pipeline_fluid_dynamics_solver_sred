# tests/grid_modules/test_grid_geometry.py
# ✅ Validation suite for src/grid_modules/grid_geometry.py

import pytest
from src.grid_modules.grid_geometry import generate_coordinates

def test_generate_coordinates_valid_domain():
    domain = {
        "min_x": 0, "max_x": 2, "nx": 2,
        "min_y": 0, "max_y": 2, "ny": 2,
        "min_z": 0, "max_z": 2, "nz": 2
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 8
    assert all(len(c) == 6 for c in coords)
    assert coords[0] == (0, 0, 0, 0.5, 0.5, 0.5)

def test_generate_coordinates_resolution_one():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    coords = generate_coordinates(domain)
    assert coords == [(0, 0, 0, 0.5, 0.5, 0.5)]

def test_generate_coordinates_missing_keys():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        # Missing z keys
    }
    with pytest.raises(ValueError) as e:
        generate_coordinates(domain)
    assert "Missing domain keys" in str(e.value)

def test_generate_coordinates_invalid_types():
    domain = {
        "min_x": "a", "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    with pytest.raises(ValueError) as e:
        generate_coordinates(domain)
    assert "Domain must contain numeric bounds" in str(e.value)

def test_generate_coordinates_zero_resolution():
    domain = {
        "min_x": 0, "max_x": 1, "nx": 0,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    with pytest.raises(ValueError) as e:
        generate_coordinates(domain)
    assert "Resolution values must be greater than zero" in str(e.value)

def test_generate_coordinates_negative_bounds():
    domain = {
        "min_x": -1, "max_x": 1, "nx": 2,
        "min_y": -2, "max_y": 2, "ny": 2,
        "min_z": -3, "max_z": 3, "nz": 2
    }
    coords = generate_coordinates(domain)
    assert coords[0] == (0, 0, 0, -0.5, -1.0, -1.5)
    assert len(coords) == 8

def test_generate_coordinates_prints_debug(capsys):
    domain = {
        "min_x": 0, "max_x": 1, "nx": 1,
        "min_y": 0, "max_y": 1, "ny": 1,
        "min_z": 0, "max_z": 1, "nz": 1
    }
    generate_coordinates(domain)
    output = capsys.readouterr().out
    assert "[GEOMETRY]" in output
    assert "Generated 1 grid coordinates" in output



