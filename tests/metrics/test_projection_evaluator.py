# tests/test_projection_evaluator.py
# 🧪 Unit tests for calculate_projection_passes — verifies velocity variability heuristic for projection depth

import pytest
import math
from src.grid_modules.cell import Cell
from src.metrics.projection_evaluator import calculate_projection_passes

def make_cell(vx, vy, vz):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=[vx, vy, vz], pressure=0.0, fluid_mask=True)

def test_empty_grid_returns_one():
    assert calculate_projection_passes([]) == 1

def test_all_none_velocity_returns_one():
    grid = [Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)]
    assert calculate_projection_passes(grid) == 1

def test_malformed_velocity_skipped():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True),
        Cell(x=0.0, y=0.0, z=0.0, velocity="invalid", pressure=0.0, fluid_mask=True)
    ]
    assert calculate_projection_passes(grid) == 1

def test_uniform_velocity_returns_one():
    grid = [make_cell(1.0, 1.0, 1.0)] * 5
    assert calculate_projection_passes(grid) == 1

def test_low_variation_returns_one():
    grid = [
        make_cell(1.0, 0.0, 0.0),  # mag = 1.0
        make_cell(1.1, 0.0, 0.0)   # mag ≈ 1.1 → variation ≈ 0.05
    ]
    assert calculate_projection_passes(grid) == 1

def test_half_unit_variation_returns_two():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        make_cell(2.0, 0.0, 0.0)
    ]
    mag1 = 1.0
    mag2 = 2.0
    avg = (mag1 + mag2) / 2
    variation = max(mag1, mag2) - avg
    assert variation > 0.5
    assert calculate_projection_passes(grid) == 2

def test_extreme_variation_returns_large_value():
    grid = [
        make_cell(1.0, 0.0, 0.0),           # mag = 1.0
        make_cell(100.0, 0.0, 0.0)          # mag = 100.0
    ]
    variation = 100.0 - 50.5  # average
    expected = 1 + int(variation // 0.5)
    assert calculate_projection_passes(grid) == expected

def test_negative_velocity_components():
    grid = [
        make_cell(-5.0, 0.0, 0.0),
        make_cell(5.0, 0.0, 0.0)
    ]
    # both have magnitude 5.0 → variation = 0.0
    assert calculate_projection_passes(grid) == 1

def test_mixed_valid_and_invalid_cells():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True),
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True)
    ]
    assert calculate_projection_passes(grid) == 1