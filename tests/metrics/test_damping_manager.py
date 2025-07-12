# tests/metrics/test_damping_manager.py

import pytest
from src.metrics.damping_manager import should_dampen
from src.grid_modules.cell import Cell

def test_uniform_velocity_no_damping():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_velocity_spike_triggers_damping():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[3.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is True

def test_low_variation_no_damping():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.1, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[1.2, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_empty_grid_returns_false():
    grid = []
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_negative_time_step_returns_false():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[2.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = -0.01
    assert should_dampen(grid, time_step) is False

def test_velocity_format_invalid():
    class BadCell:
        def __init__(self, velocity):
            self.velocity = velocity

    grid = [
        BadCell("invalid"),
        BadCell([1.0]),
        BadCell(None)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False

def test_velocity_with_yz_component():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.0, 1.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[0.0, 0.0, 2.5], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[0.0, 0.5, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is True

def test_extreme_spike_vs_average_velocity():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[0.2, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[0.2, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is True

def test_edge_case_equal_max_and_avg():
    grid = [
        Cell(x=0, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=1, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True),
        Cell(x=2, y=0, z=0, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)
    ]
    time_step = 0.1
    assert should_dampen(grid, time_step) is False



