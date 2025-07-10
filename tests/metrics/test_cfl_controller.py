# tests/metrics/test_cfl_controller.py

import pytest
from src.metrics.cfl_controller import compute_global_cfl

def test_uniform_velocity_grid():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.0, 0.0, 0.0], 1.0]
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    result = compute_global_cfl(grid, time_step, domain)
    expected_cfl = 0.1 / 0.1  # u × dt / dx = 1.0 × 0.1 / 0.1
    assert result == round(expected_cfl, 5)

def test_varied_velocity_grid():
    grid = [
        [0, 0, 0, [0.5, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.5, 0.0, 0.0], 1.0],
        [2, 0, 0, [0.1, 0.0, 0.0], 1.0]
    ]
    domain = {"nx": 20, "min_x": 0.0, "max_x": 2.0}
    time_step = 0.1
    result = compute_global_cfl(grid, time_step, domain)
    dx = 0.1
    expected_cfl = 1.5 * time_step / dx
    assert result == round(expected_cfl, 5)

def test_empty_grid_returns_zero():
    grid = []
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    result = compute_global_cfl(grid, time_step, domain)
    assert result == 0.0

def test_missing_domain_keys_returns_zero():
    grid = [[0, 0, 0, [1.0, 0.0, 0.0], 1.0]]
    incomplete_domains = [
        {"min_x": 0.0, "max_x": 1.0},
        {"nx": 10, "max_x": 1.0},
        {"nx": 10, "min_x": 0.0}
    ]
    for domain in incomplete_domains:
        assert compute_global_cfl(grid, 0.1, domain) == 0.0

def test_invalid_velocity_format():
    grid = [
        [0, 0, 0, "not a vector", 1.0],
        [1, 0, 0, [1.0, 0.0], 1.0],
        [2, 0, 0, None, 1.0]
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    result = compute_global_cfl(grid, time_step, domain)
    assert result == 0.0

def test_velocity_with_yz_components():
    grid = [
        [0, 0, 0, [0.0, 1.0, 0.0], 1.0],
        [1, 0, 0, [0.0, 0.0, 2.0], 1.0]
    ]
    domain = {"nx": 10, "min_x": 0.0, "max_x": 1.0}
    time_step = 0.1
    dx = 0.1
    expected_magnitude = max((1.0**2)**0.5, (2.0**2)**0.5)
    expected_cfl = expected_magnitude * time_step / dx
    result = compute_global_cfl(grid, time_step, domain)
    assert result == round(expected_cfl, 5)



