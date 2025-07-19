# tests/metrics/test_damping_manager.py
# 🧪 Unit tests for src/metrics/damping_manager.py

from src.grid_modules.cell import Cell
from src.metrics.damping_manager import should_dampen

def make_cell(v):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=v, pressure=0.0, fluid_mask=True)

def test_should_dampen_triggered_by_spike():
    # One high spike compared to the average of others
    grid = [make_cell([1.0, 0.0, 0.0]), make_cell([1.0, 0.0, 0.0]), make_cell([3.0, 0.0, 0.0])]
    result = should_dampen(grid, time_step=0.1)
    assert result is True

def test_should_dampen_not_triggered_if_velocities_uniform():
    grid = [make_cell([1.0, 0.0, 0.0]), make_cell([1.0, 0.0, 0.0]), make_cell([1.0, 0.0, 0.0])]
    result = should_dampen(grid, time_step=0.1)
    assert result is False

def test_should_dampen_returns_false_if_time_step_zero():
    grid = [make_cell([2.0, 0.0, 0.0])]
    result = should_dampen(grid, time_step=0.0)
    assert result is False

def test_should_dampen_returns_false_if_grid_empty():
    result = should_dampen([], time_step=0.1)
    assert result is False

def test_should_dampen_handles_non_vector_velocity():
    grid = [make_cell(None), make_cell([1.0]), make_cell("bad")]
    result = should_dampen(grid, time_step=0.1)
    assert result is False

def test_should_dampen_returns_false_if_no_valid_magnitudes():
    # All cells invalid
    grid = [make_cell(None), make_cell([])]
    result = should_dampen(grid, time_step=0.1)
    assert result is False

def test_should_dampen_edge_case_exactly_50_percent():
    # Average = 1.0, spike = 1.5 → volatility = 0.5 → not greater than 0.5 * avg
    grid = [make_cell([1.0, 0.0, 0.0]), make_cell([1.0, 0.0, 0.0]), make_cell([1.5, 0.0, 0.0])]
    result = should_dampen(grid, time_step=0.1)
    assert result is False



