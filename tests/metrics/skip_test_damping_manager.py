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

    grid[2].velocity = [2.0, 0.0, 0.0]  # ✅ clearly above dynamic threshold
    assert should_dampen(grid, time_step=0.1) is True

@pytest.mark.parametrize("spike_velocity,expected", [
    ([1.5, 0.0, 0.0], False),
    ([2.0, 0.0, 0.0], True),
    ([3.0, 0.0, 0.0], True),
    ([5.0, 0.0, 0.0], True)
])
def test_damping_threshold_sweep(spike_velocity, expected):
    grid = [
        mock_cell(0, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(1, 0, 0, [1.0, 0.0, 0.0]),
        mock_cell(2, 0, 0, spike_velocity)
    ]
    assert should_dampen(grid, time_step=0.1) is expected

def test_damping_threshold_documentation():
    """
    Documents volatility threshold behavior for damping logic.
    Ensures that velocity spikes above baseline trigger damping,
    while near-uniform fields do not.
    """
    baseline = [1.0, 0.0, 0.0]
    spike = [2.0, 0.0, 0.0]
    grid = [
        mock_cell(0, 0, 0, baseline),
        mock_cell(1, 0, 0, baseline),
        mock_cell(2, 0, 0, spike)
    ]
    assert should_dampen(grid, time_step=0.1) is True



