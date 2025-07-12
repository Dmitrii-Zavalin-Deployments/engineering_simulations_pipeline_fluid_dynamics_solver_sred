# tests/metrics/test_divergence_metrics.py

import pytest
from src.metrics.divergence_metrics import compute_max_divergence
from src.grid_modules.cell import Cell

def test_uniform_velocity_no_divergence():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 0.0

def test_linear_velocity_gradient_x_direction():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[4.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 2.0  # |4.0 - 2.0|

def test_negative_velocity_gradient():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.5, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[0.5, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 1.5  # |1.5 - 3.0|

def test_non_monotonic_velocity_pattern():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.5, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 2.5  # |3.0 - 0.5|

def test_short_grid_returns_zero():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 0.0

def test_empty_grid_returns_zero():
    result = compute_max_divergence([])
    assert result == 0.0

def test_invalid_velocity_formats_ignored():
    class BadCell:
        def __init__(self, velocity):
            self.velocity = velocity

    grid = [
        BadCell("invalid"),
        BadCell([1.0]),
        BadCell(None),
        Cell(x=3, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=4, y=0, z=0, velocity=[5.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 3.0  # |5.0 - 2.0|

def test_fluctuating_velocity_components():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 1.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[0.5, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    result = compute_max_divergence(grid)
    assert result == 1.5  # |0.5 - 2.0|



