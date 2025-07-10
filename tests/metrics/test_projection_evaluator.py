# tests/metrics/test_projection_evaluator.py

import pytest
from src.metrics.projection_evaluator import calculate_projection_passes

def test_uniform_velocity_returns_one():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.0, 0.0, 0.0], 1.0]
    ]
    assert calculate_projection_passes(grid) == 1

def test_linear_velocity_gradient():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [2.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [3.0, 0.0, 0.0], 1.0]
    ]
    # max_v = 3.0, avg_v = 2.0, variation = 1.0 → passes = 1 + int(1.0 // 0.5) = 3
    assert calculate_projection_passes(grid) == 3

def test_spike_in_velocity():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [5.0, 0.0, 0.0], 1.0]
    ]
    # max_v = 5.0, avg ≈ 2.33, variation ≈ 2.67 → passes = 6
    assert calculate_projection_passes(grid) == 6

def test_empty_grid_returns_one():
    assert calculate_projection_passes([]) == 1

def test_invalid_velocity_entries_skipped():
    grid = [
        [0, 0, 0, "invalid", 1.0],
        [1, 0, 0, None, 1.0],
        [2, 0, 0, [2.0], 1.0],
        [3, 0, 0, [3.0, 0.0, 0.0], 1.0]
    ]
    # Only one valid velocity → max = avg = variation = 0 → passes = 1
    assert calculate_projection_passes(grid) == 1

def test_high_variation_with_yz_components():
    grid = [
        [0, 0, 0, [1.0, 2.0, 2.0], 1.0],  # mag ≈ 3.0
        [1, 0, 0, [0.0, 0.0, 0.0], 1.0],  # mag = 0
        [2, 0, 0, [2.0, 0.0, 0.0], 1.0]   # mag = 2.0
    ]
    # max = 3.0, avg ≈ 1.67, variation ≈ 1.33 → passes = 1 + int(1.33 // 0.5) = 3
    assert calculate_projection_passes(grid) == 3

def test_high_velocity_burst_many_cells():
    grid = [[i, 0, 0, [i % 2 * 10.0, 0.0, 0.0], 1.0] for i in range(10)]
    # High variability → should trigger higher number of passes
    result = calculate_projection_passes(grid)
    assert result >= 10  # Depending on burst layout, variation should be large



