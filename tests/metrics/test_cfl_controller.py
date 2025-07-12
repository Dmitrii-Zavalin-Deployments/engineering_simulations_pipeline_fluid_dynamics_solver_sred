# tests/metrics/test_cfl_controller.py

import pytest
from src.metrics.cfl_controller import compute_global_cfl
from src.grid_modules.cell import Cell

def test_uniform_velocity_grid():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    result = compute_global_cfl(grid, time_step, domain)
    expected_cfl = 1.0 * time_step / (1.0 / domain["nx"])
    assert result == round(expected_cfl, 5)

def test_varied_velocity_grid():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.5, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.5, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[0.1, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    domain = {"nx": 20, "min_x": 0.0, "max_x": 2.0}
    time_step = 0.1
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    expected_cfl = 1.5 * time_step / dx
    result = compute_global_cfl(grid, time_step, domain)
    assert result == round(expected_cfl, 5)

def test_empty_grid_returns_zero():
    grid = []
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    assert compute_global_cfl(grid, time_step, domain) == 0.0

def test_missing_domain_keys_returns_zero():
    grid = [Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)]
    incomplete_domains = [
        {"min_x": 0.0, "max_x": 1.0},
        {"nx": 10, "max_x": 1.0},
        {"nx": 10, "min_x": 0.0}
    ]
    for domain in incomplete_domains:
        assert compute_global_cfl(grid, 0.1, domain) == 0.0

def test_invalid_velocity_format():
    class BadCell:
        def __init__(self, velocity):
            self.velocity = velocity

    grid = [
        BadCell("not a vector"),
        BadCell([1.0, 0.0]),  # wrong length
        BadCell(None)
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    result = compute_global_cfl(grid, time_step, domain)
    assert result == 0.0

def test_velocity_with_yz_components():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.0, 1.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[0.0, 0.0, 2.0], pressure=1.0, fluid_mask=True)
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    expected_magnitude = 2.0
    expected_cfl = expected_magnitude * time_step / dx
    result = compute_global_cfl(grid, time_step, domain)
    assert result == round(expected_cfl, 5)



