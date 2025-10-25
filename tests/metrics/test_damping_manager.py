# tests/metrics/test_damping_manager.py
# ✅ Validation suite for src/metrics/damping_manager.py

import pytest
from src.metrics.damping_manager import should_dampen
from src.grid_modules.cell import Cell

def mock_cell(x, y, z, velocity, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid_mask)

def test_damping_triggered_by_high_volatility():
    grid = [
        mock_cell(0, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(1, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(2, 0, 0, [5.0, 0.0, 0.0])  # spike
    ]
    assert should_dampen(grid, time_step=0.1) is True

def test_damping_not_triggered_by_uniform_velocity():
    grid = [
        mock_cell(0, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(1, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(2, 0, 0, [1.0, 0.0, 0.0])
    ]
    assert should_dampen(grid, time_step=0.1) is False

def test_damping_excludes_solid_cells():
    grid = [
        mock_cell(0, 0, 0, [10.0, 0.0, 0.0], fluid_mask=False),  # excluded
        mock_cell(1, 0, 0, [1.0, 0.0, 0.0], fluid_mask=True),
        mock_cell(2, 0, 0, [1.0, 0.0, 0.0], fluid_mask=True)
    ]
    assert should_dampen(grid, time_step=0.1) is False

def test_damping_handles_missing_velocity():
    grid = [
        mock_cell(0, 0, 0, None),
        mock_cell(1, 0, 0, [1.0, 1.0])  # malformed
    ]
    assert should_dampen(grid, time_step=0.1) is False

def test_damping_handles_empty_grid():
    assert should_dampen([], time_step=0.1) is False

def test_damping_handles_zero_time_step():
    grid = [mock_cell(0, 0, 0, [1.0, 0.0, 0.0])]
    assert should_dampen(grid, time_step=0.0) is False

def test_damping_handles_negative_time_step():
    grid = [mock_cell(0, 0, 0, [1.0, 0.0, 0.0])]
    assert should_dampen(grid, time_step=-0.1) is False

def test_damping_precision_near_threshold():
    grid = [
        mock_cell(0, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(1, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(2, 0, 0, [1.49, 0.0, 0.0])  # just below trigger
    ]
    assert should_dampen(grid, time_step=0.1) is False

    grid[2].velocity = [1.51, 0.0, 0.0]  # just above trigger
    assert should_dampen(grid, time_step=0.1) is True



