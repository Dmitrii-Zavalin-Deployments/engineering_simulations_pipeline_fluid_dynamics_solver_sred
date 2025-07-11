# tests/grid/test_grid_geometry.py

import pytest
from src.grid_modules.grid_geometry import generate_coordinates

# ✅ Test: Regular domain resolution produces correct number of points
def test_coordinate_count_matches_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 2,
        "min_z": 0.0, "max_z": 1.0, "nz": 2
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 2 * 2 * 2
    expected = [
        (0.25, 0.25, 0.25), (0.25, 0.25, 0.75),
        (0.25, 0.75, 0.25), (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.25), (0.75, 0.25, 0.75),
        (0.75, 0.75, 0.25), (0.75, 0.75, 0.75),
    ]
    assert coords == expected

# ✅ Test: Zero-sized grid returns empty list
def test_zero_resolution_grid():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 0,
        "min_y": 0.0, "max_y": 1.0, "ny": 0,
        "min_z": 0.0, "max_z": 1.0, "nz": 0
    }
    coords = generate_coordinates(domain)
    assert coords == []

# ✅ Test: Negative bounds still compute correctly
def test_negative_domain_bounds():
    domain = {
        "min_x": -1.0, "max_x": 1.0, "nx": 2,
        "min_y": -2.0, "max_y": 2.0, "ny": 2,
        "min_z": -3.0, "max_z": 3.0, "nz": 2
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 8
    assert coords[0] == ( -0.5, -1.0, -1.5 )
    assert coords[-1] == (  0.5,  1.0,  1.5 )

# ✅ Test: Uneven resolution along axes
def test_nonuniform_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 3
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 2 * 1 * 3
    assert coords[0] == (0.25, 0.5, 0.16666666666666666)
    assert coords[-1] == (0.75, 0.5, 0.8333333333333334)

# ❌ Test: Missing domain keys raises KeyError
def test_missing_domain_keys():
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 2}  # Missing y/z info
    with pytest.raises(KeyError):
        generate_coordinates(domain)

# ❌ Test: Resolution as non-integer triggers TypeError
def test_non_integer_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": "two",
        "min_y": 0.0, "max_y": 1.0, "ny": 2,
        "min_z": 0.0, "max_z": 1.0, "nz": 2
    }
    with pytest.raises(TypeError):
        generate_coordinates(domain)



