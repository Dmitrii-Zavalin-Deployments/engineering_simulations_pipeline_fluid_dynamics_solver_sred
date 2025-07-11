# tests/metrics/test_projection_evaluator.py

import pytest
from src.metrics.projection_evaluator import calculate_projection_passes
from src.grid_modules.cell import Cell

def test_uniform_velocity_returns_one():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0),
        Cell(x=2, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0)
    ]
    assert calculate_projection_passes(grid) == 1

def test_linear_velocity_gradient():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0),
        Cell(x=1, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0),
        Cell(x=2, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0)
    ]
    assert calculate_projection_passes(grid) == 3

def test_spike_in_velocity():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0),
        Cell(x=2, y=0, z=0, velocity=[5.0, 0.0, 0.0], pressure=1.0)
    ]
    assert calculate_projection_passes(grid) == 6

def test_empty_grid_returns_one():
    assert calculate_projection_passes([]) == 1

def test_invalid_velocity_entries_skipped():
    class BadCell:
        def __init__(self, velocity):
            self.velocity = velocity

    grid = [
        BadCell("invalid"),
        BadCell(None),
        BadCell([2.0]),
        Cell(x=3, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0)
    ]
    assert calculate_projection_passes(grid) == 1

def test_high_variation_with_yz_components():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 2.0, 2.0], pressure=1.0),  # mag ≈ 3.0
        Cell(x=1, y=0, z=0, velocity=[0.0, 0.0, 0.0], pressure=1.0),  # mag = 0
        Cell(x=2, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0)   # mag = 2.0
    ]
    assert calculate_projection_passes(grid) == 3

def test_high_velocity_burst_many_cells():
    grid = [
        Cell(x=i, y=0, z=0, velocity=[(i % 2) * 10.0, 0.0, 0.0], pressure=1.0)
        for i in range(10)
    ]
    result = calculate_projection_passes(grid)
    assert result >= 10



