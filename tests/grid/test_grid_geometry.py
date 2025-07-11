# tests/grid/test_grid_geometry.py

import pytest
import math
from src.grid_modules.grid_geometry import generate_coordinates

# ✅ Test: Regular domain resolution produces correct number of points
def test_coordinate_count_matches_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 2,
        "min_z": 0.0, "max_z": 1.0, "nz": 2
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 8
    expected = [
        (0.25, 0.25, 0.25), (0.25, 0.25, 0.75),
        (0.25, 0.75, 0.25), (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.25), (0.75, 0.25, 0.75),
        (0.75, 0.75, 0.25), (0.75, 0.75, 0.75),
    ]
    for a, e in zip(coords, expected):
        assert all(math.isclose(a_i, e_i, rel_tol=1e-9) for a_i, e_i in zip(a, e))

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
    expected_first = (-0.5, -1.0, -1.5)
    expected_last = ( 0.5,  1.0,  1.5)
    for a_i, e_i in zip(coords[0], expected_first):
        assert math.isclose(a_i, e_i, rel_tol=1e-9)
    for a_i, e_i in zip(coords[-1], expected_last):
        assert math.isclose(a_i, e_i, rel_tol=1e-9)

# ✅ Test: Uneven resolution along axes
def test_nonuniform_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": 2,
        "min_y": 0.0, "max_y": 1.0, "ny": 1,
        "min_z": 0.0, "max_z": 1.0, "nz": 3
    }
    coords = generate_coordinates(domain)
    assert len(coords) == 6
    expected_first = (0.25, 0.5, 0.16666666666666666)
    expected_last  = (0.75, 0.5, 0.8333333333333334)
    for a_i, e_i in zip(coords[0], expected_first):
        assert math.isclose(a_i, e_i, rel_tol=1e-9)
    for a_i, e_i in zip(coords[-1], expected_last):
        assert math.isclose(a_i, e_i, rel_tol=1e-9)

# ❌ Test: Missing domain keys raises ValueError
def test_missing_domain_keys():
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 2}  # Missing y/z info
    with pytest.raises(ValueError):
        generate_coordinates(domain)

# ❌ Test: Resolution as non-integer triggers ValueError
def test_non_integer_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0, "nx": "two",
        "min_y": 0.0, "max_y": 1.0, "ny": 2,
        "min_z": 0.0, "max_z": 1.0, "nz": 2
    }
    with pytest.raises(ValueError):
        generate_coordinates(domain)



