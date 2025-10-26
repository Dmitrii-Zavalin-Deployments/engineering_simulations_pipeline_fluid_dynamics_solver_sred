import pytest
import math
from src.metrics.overflow_monitor import detect_overflow
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, fluid_mask=True):
    cell = Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=0.0,
        fluid_mask=fluid_mask
    )
    cell.overflow_triggered = False  # ensure flag is present
    return cell

@pytest.fixture
def low_velocity_grid():
    return [
        make_cell(0.0, 0.0, 0.0, velocity=[0.1, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.2, 0.0, 0.0])
    ]

@pytest.fixture
def high_velocity_grid():
    return [
        make_cell(0.0, 0.0, 0.0, velocity=[1e20, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[1e22, 0.0, 0.0])
    ]

def test_detect_overflow_returns_false_for_empty_grid():
    assert detect_overflow([]) is False
    assert detect_overflow(None) is False

def test_detect_overflow_excludes_non_fluid_cells():
    cell_solid = make_cell(0.0, 0.0, 0.0, velocity=[100.0, 0.0, 0.0], fluid_mask=False)
    assert detect_overflow([cell_solid]) is False
    assert not cell_solid.overflow_triggered

def test_detect_overflow_returns_false_for_safe_velocities(low_velocity_grid):
    result = detect_overflow(low_velocity_grid)
    assert result is False
    assert all(not cell.overflow_triggered for cell in low_velocity_grid)

def test_detect_overflow_returns_true_for_extreme_velocities(high_velocity_grid):
    result = detect_overflow(high_velocity_grid)
    assert result is True
    assert any(cell.overflow_triggered for cell in high_velocity_grid)

def test_detect_overflow_does_not_mutate_unaffected_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1e20, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.1, 0.0, 0.0])
    ]
    result = detect_overflow(grid)
    assert result is True
    assert grid[0].overflow_triggered is True
    assert grid[1].overflow_triggered is False

def test_detect_overflow_handles_malformed_velocity_gracefully():
    malformed_1 = make_cell(0.0, 0.0, 0.0, velocity=None)
    malformed_2 = make_cell(1.0, 1.0, 1.0, velocity=[1.0])  # too short
    malformed_3 = make_cell(2.0, 2.0, 2.0, velocity="fast")  # wrong type
    assert detect_overflow([malformed_1, malformed_2, malformed_3]) is False

def test_detect_overflow_threshold_boundary():
    edge_cell = make_cell(0.0, 0.0, 0.0, velocity=[6.0, 8.0, 0.0])  # magnitude = 10.0
    spike_cell = make_cell(1.0, 1.0, 1.0, velocity=[6.1, 8.0, 0.0])  # magnitude > 10.0

    assert math.isclose(math.sqrt(6.0**2 + 8.0**2), 10.0)
    assert detect_overflow([edge_cell]) is False
    assert detect_overflow([spike_cell]) is True

def test_detect_overflow_threshold_documentation():
    """
    Explicitly documents the overflow threshold used in detect_overflow.
    This test ensures that velocity magnitude just below the threshold does not trigger overflow,
    while values just above it do.
    """
    threshold = 10.0

    below = make_cell(0.0, 0.0, 0.0, velocity=[5.9, 8.0, 0.0])  # magnitude ≈ 9.99
    above = make_cell(1.0, 1.0, 1.0, velocity=[6.1, 8.0, 0.0])  # magnitude ≈ 10.02

    assert math.sqrt(sum(v**2 for v in below.velocity)) < threshold
    assert math.sqrt(sum(v**2 for v in above.velocity)) > threshold

    assert detect_overflow([below]) is False
    assert detect_overflow([above]) is True



