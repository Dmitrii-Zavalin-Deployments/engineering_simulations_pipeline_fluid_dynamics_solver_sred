# tests/metrics/test_damping_manager.py

import pytest
from src.metrics.damping_manager import should_dampen

def test_uniform_velocity_no_damping():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.0, 0.0, 0.0], 1.0]
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_velocity_spike_triggers_damping():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [3.0, 0.0, 0.0], 1.0]  # Spike
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is True

def test_low_variation_no_damping():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.1, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.2, 0.0, 0.0], 1.0]
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_empty_grid_returns_false():
    grid = []
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_negative_time_step_returns_false():
    grid = [[0, 0, 0, [2.0, 0.0, 0.0], 1.0]]
    time_step = -0.01
    assert should_dampen(grid, time_step) is False

def test_velocity_format_invalid():
    grid = [
        [0, 0, 0, "invalid", 1.0],
        [1, 0, 0, [1.0], 1.0],
        [2, 0, 0, None, 1.0]
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_velocity_with_yz_component():
    grid = [
        [0, 0, 0, [0.0, 1.0, 0.0], 1.0],
        [1, 0, 0, [0.0, 0.0, 2.5], 1.0],
        [2, 0, 0, [0.0, 0.5, 0.0], 1.0]
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is True

def test_extreme_spike_vs_average_velocity():
    grid = [
        [0, 0, 0, [0.2, 0.0, 0.0], 1.0],
        [1, 0, 0, [0.2, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.0, 0.0, 0.0], 1.0]
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is True

def test_edge_case_equal_max_and_avg():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.0, 0.0, 0.0], 1.0]
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False



