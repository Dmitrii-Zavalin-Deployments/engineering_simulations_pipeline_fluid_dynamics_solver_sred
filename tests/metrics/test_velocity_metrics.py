# tests/metrics/test_velocity_metrics.py

import pytest
from src.metrics.velocity_metrics import compute_max_velocity
from src.grid_modules.cell import Cell

def test_uniform_velocity_returns_correct_magnitude():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0)
    ]
    assert compute_max_velocity(grid) == 1.0

def test_varied_velocity_components():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 2.0, 2.0], pressure=1.0),  # magnitude ≈ 3.0
        Cell(x=1, y=0, z=0, velocity=[0.0, 3.0, 4.0], pressure=1.0),  # magnitude = 5.0
        Cell(x=2, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0)   # magnitude = 2.0
    ]
    assert compute_max_velocity(grid) == 5.0

def test_empty_grid_returns_zero():
    assert compute_max_velocity([]) == 0.0

def test_malformed_velocity_vectors():
    class BadCell:
        def __init__(self, velocity):
            self.velocity = velocity

    grid = [
        BadCell("invalid"),
        BadCell(None),
        BadCell([1.0]),
        Cell(x=3, y=0, z=0, velocity=[5.0, 0.0, 0.0], pressure=1.0)
    ]
    assert compute_max_velocity(grid) == 5.0

def test_negative_velocity_components():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[-3.0, 0.0, 0.0], pressure=1.0),     # magnitude = 3.0
        Cell(x=1, y=0, z=0, velocity=[-4.0, -3.0, 0.0], pressure=1.0)     # magnitude = 5.0
    ]
    assert compute_max_velocity(grid) == 5.0

def test_zero_velocity_magnitude():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.0, 0.0, 0.0], pressure=1.0),
        Cell(x=1, y=0, z=0, velocity=[0.0, 0.0, 0.0], pressure=1.0)
    ]
    assert compute_max_velocity(grid) == 0.0

def test_high_precision_magnitude_rounding():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.234567, 2.345678, 3.456789], pressure=1.0)
    ]
    expected = round((1.234567**2 + 2.345678**2 + 3.456789**2) ** 0.5, 5)
    assert compute_max_velocity(grid) == expected



