# tests/utils/test_grid_spacing.py

import pytest
from src.utils.grid_spacing import compute_grid_spacing

def test_compute_grid_spacing_returns_correct_values():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 2.0,
        "min_z": 0.0, "max_z": 4.0,
        "nx": 4, "ny": 4, "nz": 4
    }
    dx, dy, dz = compute_grid_spacing(domain)
    assert dx == pytest.approx(0.25)
    assert dy == pytest.approx(0.5)
    assert dz == pytest.approx(1.0)

def test_compute_grid_spacing_handles_nonzero_min_bounds():
    domain = {
        "min_x": 1.0, "max_x": 5.0,
        "min_y": -2.0, "max_y": 2.0,
        "min_z": 10.0, "max_z": 20.0,
        "nx": 4, "ny": 4, "nz": 5
    }
    dx, dy, dz = compute_grid_spacing(domain)
    assert dx == pytest.approx(1.0)
    assert dy == pytest.approx(1.0)
    assert dz == pytest.approx(2.0)

def test_compute_grid_spacing_raises_on_missing_keys():
    incomplete_domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        # Missing min_z, max_z, nz
        "nx": 2, "ny": 2
    }
    with pytest.raises(KeyError):
        compute_grid_spacing(incomplete_domain)

def test_compute_grid_spacing_raises_on_zero_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 0, "ny": 2, "nz": 2
    }
    with pytest.raises(ZeroDivisionError):
        compute_grid_spacing(domain)
