# tests/metrics/test_overflow_monitor.py

import pytest
from src.metrics.overflow_monitor import detect_overflow
from src.grid_modules.cell import Cell

def test_no_overflow_uniform_velocity():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.0, 1.0, 1.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    assert detect_overflow(grid) is False

def test_detects_overflow_single_cell():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[12.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),  # Exceeds threshold
        Cell(x=2, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    assert detect_overflow(grid) is True

def test_overflow_on_magnitude_combination():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[6.0, 8.0, 0.0], pressure=1.0, fluid_mask=True)  # Magnitude = 10.0
    ]
    assert detect_overflow(grid) is False  # Edge case: exactly at threshold

    grid_with_spike = [
        Cell(x=0, y=0, z=0, velocity=[6.0, 8.1, 0.0], pressure=1.0, fluid_mask=True)  # Magnitude > 10.0
    ]
    assert detect_overflow(grid_with_spike) is True

def test_empty_grid_returns_false():
    assert detect_overflow([]) is False

def test_malformed_velocity_vector_handled():
    class BadCell:
        def __init__(self, velocity):
            self.velocity = velocity

    grid = [
        BadCell([1.0, 2.0]),
        BadCell("invalid"),
        BadCell(None),
        Cell(x=3, y=0, z=0, velocity=[9.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    assert detect_overflow(grid) is False

def test_multiple_cells_with_overflow():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[11.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[0.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[10.1, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    assert detect_overflow(grid) is True

def test_negative_velocity_components():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[-8.0, -6.0, 0.0], pressure=1.0, fluid_mask=True)  # Magnitude = 10.0
    ]
    assert detect_overflow(grid) is False

    grid_spike = [
        Cell(x=0, y=0, z=0, velocity=[-8.0, -6.1, 0.0], pressure=1.0, fluid_mask=True)  # Magnitude > 10.0
    ]
    assert detect_overflow(grid_spike) is True



