# tests/test_cfl_controller.py
# 🧪 Unit tests for CFL calculation — verifies CFL number across velocity magnitudes and domain configs

import pytest
from src.grid_modules.cell import Cell
from src.metrics.cfl_controller import compute_global_cfl

def make_cell(vx, vy, vz):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=[vx, vy, vz], pressure=0.0, fluid_mask=True)

def test_basic_cfl_computation():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        make_cell(0.0, 2.0, 0.0),
        make_cell(0.0, 0.0, 3.0)
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.01
    cfl = compute_global_cfl(grid, time_step, domain)
    expected_magnitude = 3.0
    dx = 0.1
    expected_cfl = round(expected_magnitude * time_step / dx, 5)
    assert cfl == expected_cfl

def test_empty_grid_returns_zero():
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.01
    assert compute_global_cfl([], time_step, domain) == 0.0

@pytest.mark.parametrize("missing_key", ["nx", "min_x", "max_x"])
def test_missing_domain_keys_returns_zero(missing_key):
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    del domain[missing_key]
    grid = [make_cell(1.0, 0.0, 0.0)]
    assert compute_global_cfl(grid, 0.01, domain) == 0.0

def test_cell_with_none_velocity_skipped():
    bad_cell = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)
    grid = [bad_cell, make_cell(0.0, 0.0, 2.0)]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    cfl = compute_global_cfl(grid, 0.01, domain)
    expected = round(2.0 * 0.01 / 0.1, 5)
    assert cfl == expected

def test_cell_with_invalid_velocity_shape_skipped():
    bad_cell = Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True)
    grid = [bad_cell, make_cell(0.0, 0.0, 1.0)]
    domain = {"nx": 5, "min_x": 0.0, "max_x": 1.0}
    cfl = compute_global_cfl(grid, 0.02, domain)
    expected = round(1.0 * 0.02 / 0.2, 5)
    assert cfl == expected

def test_negative_velocity_values():
    grid = [
        make_cell(-1.0, -1.0, -1.0),
        make_cell(0.0, 0.0, -2.0)
    ]
    domain = {"nx": 4, "min_x": 0.0, "max_x": 2.0}
    time_step = 0.05
    magnitude = max(
        (3**0.5),  # sqrt(1^2 + 1^2 + 1^2)
        abs(2.0)
    )
    dx = 0.5
    expected_cfl = round(magnitude * time_step / dx, 5)
    assert compute_global_cfl(grid, time_step, domain) == expected_cfl

def test_large_grid_max_cfl():
    grid = [make_cell(0.1 * i, 0.2 * i, 0.3 * i) for i in range(10)]
    domain = {"nx": 100, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.01
    last_magnitude = ( (0.9)**2 + (1.8)**2 + (2.7)**2 ) ** 0.5
    expected = round(last_magnitude * time_step / 0.01, 5)
    assert compute_global_cfl(grid, time_step, domain) == expected