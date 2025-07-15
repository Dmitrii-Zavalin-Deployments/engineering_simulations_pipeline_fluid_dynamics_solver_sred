# tests/test_damping_manager.py
# 🧪 Unit tests for damping_manager.py — validates flow damping triggers based on velocity volatility

import pytest
from src.grid_modules.cell import Cell
from src.metrics.damping_manager import should_dampen

def make_cell(vx, vy, vz):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=[vx, vy, vz], pressure=0.0, fluid_mask=True)

def test_damping_triggers_on_velocity_spike():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        make_cell(1.0, 0.0, 0.0),
        make_cell(3.0, 0.0, 0.0)
    ]
    assert should_dampen(grid, time_step=0.01) is True

def test_damping_does_not_trigger_on_uniform_flow():
    grid = [make_cell(2.0, 0.0, 0.0)] * 5
    assert should_dampen(grid, time_step=0.01) is False

def test_empty_grid_returns_false():
    assert should_dampen([], time_step=0.01) is False

def test_zero_time_step_returns_false():
    grid = [make_cell(1.0, 0.0, 0.0)]
    assert should_dampen(grid, time_step=0.0) is False

def test_none_velocity_skipped():
    bad = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)
    good = make_cell(2.0, 2.0, 2.0)
    assert should_dampen([bad, good], time_step=0.01) is False

def test_malformed_velocity_skipped():
    bad = Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True)
    good = make_cell(1.0, 1.0, 1.0)
    assert should_dampen([bad, good], time_step=0.01) is False

def test_low_spread_does_not_trigger():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        make_cell(1.1, 0.0, 0.0),
        make_cell(0.9, 0.0, 0.0)
    ]
    assert should_dampen(grid, time_step=0.01) is False

def test_extreme_magnitude_triggers_damping():
    grid = [make_cell(1.0, 1.0, 1.0), make_cell(10.0, 0.0, 0.0)]
    assert should_dampen(grid, time_step=0.01) is True

def test_single_cell_never_triggers_damping():
    assert should_dampen([make_cell(5.0, 0.0, 0.0)], time_step=0.01) is False