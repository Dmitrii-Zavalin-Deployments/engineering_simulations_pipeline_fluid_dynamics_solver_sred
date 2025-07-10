# tests/metrics/test_divergence_metrics.py

import pytest
from src.metrics.divergence_metrics import compute_max_divergence

def test_uniform_velocity_no_divergence():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [1.0, 0.0, 0.0], 1.0]
    ]
    result = compute_max_divergence(grid)
    assert result == 0.0

def test_linear_velocity_gradient_x_direction():
    grid = [
        [0, 0, 0, [1.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [2.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [4.0, 0.0, 0.0], 1.0]
    ]
    # Max divergence = |4.0 - 2.0| = 2.0
    result = compute_max_divergence(grid)
    assert result == 2.0

def test_negative_velocity_gradient():
    grid = [
        [0, 0, 0, [3.0, 0.0, 0.0], 1.0],
        [1, 0, 0, [1.5, 0.0, 0.0], 1.0],
        [2, 0, 0, [0.5, 0.0, 0.0], 1.0]
    ]
    # Divergence = |1.5 - 3.0| = 1.5, |0.5 - 1.5| = 1.0 → max = 1.5
    result = compute_max_divergence(grid)
    assert result == 1.5

def test_non_monotonic_velocity_pattern():
    grid = [
        [0, 0, 0, [0.5, 0.0, 0.0], 1.0],
        [1, 0, 0, [3.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [2.0, 0.0, 0.0], 1.0]
    ]
    # Divergences = |3.0 - 0.5| = 2.5, |2.0 - 3.0| = 1.0 → max = 2.5
    result = compute_max_divergence(grid)
    assert result == 2.5

def test_short_grid_returns_zero():
    grid = [[0, 0, 0, [1.0, 0.0, 0.0], 1.0]]
    result = compute_max_divergence(grid)
    assert result == 0.0

def test_empty_grid_returns_zero():
    result = compute_max_divergence([])
    assert result == 0.0

def test_invalid_velocity_formats_ignored():
    grid = [
        [0, 0, 0, "invalid", 1.0],
        [1, 0, 0, [1.0], 1.0],
        [2, 0, 0, None, 1.0],
        [3, 0, 0, [2.0, 0.0, 0.0], 1.0],
        [4, 0, 0, [5.0, 0.0, 0.0], 1.0]
    ]
    # Only last two cells are valid → divergence = |5.0 - 2.0| = 3.0
    result = compute_max_divergence(grid)
    assert result == 3.0

def test_fluctuating_velocity_components():
    grid = [
        [0, 0, 0, [1.0, 1.0, 0.0], 1.0],
        [1, 0, 0, [2.0, 0.0, 0.0], 1.0],
        [2, 0, 0, [0.5, 0.0, 0.0], 1.0]
    ]
    # Only x-component analyzed → |2.0 - 1.0| = 1.0, |0.5 - 2.0| = 1.5
    result = compute_max_divergence(grid)
    assert result == 1.5



