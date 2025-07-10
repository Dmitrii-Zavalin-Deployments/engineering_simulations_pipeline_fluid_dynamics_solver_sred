# tests/metrics/test_overflow_monitor.py

import pytest
from src.metrics.overflow_monitor import detect_overflow

def test_no_overflow_uniform_velocity():
    grid = [
        [0, 0, 0, [2.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 1.0, 1.0], 1.0],
        [2, 0, 0, [3.0, 0.0, 0.0], 1.0]
    ]
    assert detect_overflow(grid) is False

def test_detects_overflow_single_cell():
    grid = [
        [0, 0, 0, [2.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [12.0, 0.0, 0.0], 1.0],  # Exceeds threshold
        [2, 0, 0, [3.0, 0.0, 0.0], 1.0]
    ]
    assert detect_overflow(grid) is True

def test_overflow_on_magnitude_combination():
    grid = [
        [0, 0, 0, [6.0, 8.0, 0.0], 1.0]  # Magnitude = 10.0
    ]
    assert detect_overflow(grid) is False  # Edge case: exactly at threshold

    grid_with_spike = [
        [0, 0, 0, [6.0, 8.1, 0.0], 1.0]  # Magnitude > 10.0
    ]
    assert detect_overflow(grid_with_spike) is True

def test_empty_grid_returns_false():
    assert detect_overflow([]) is False

def test_malformed_velocity_vector_handled():
    grid = [
        [0, 0, 0, [1.0, 2.0], 1.0],
        [1, 0, 0, "invalid", 1.0],
        [2, 0, 0, None, 1.0],
        [3, 0, 0, [9.0, 0.0, 0.0], 1.0]
    ]
    assert detect_overflow(grid) is False

def test_multiple_cells_with_overflow():
    grid = [
        [0, 0, 0, [11.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [0.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [10.1, 0.0, 0.0], 1.0]
    ]
    assert detect_overflow(grid) is True

def test_negative_velocity_components():
    grid = [
        [0, 0, 0, [-8.0, -6.0, 0.0], 1.0]  # Magnitude = 10.0
    ]
    assert detect_overflow(grid) is False

    grid_spike = [
        [0, 0, 0, [-8.0, -6.1, 0.0], 1.0]  # Magnitude > 10.0
    ]
    assert detect_overflow(grid_spike) is True



