# tests/test_velocity_metrics.py
# 🧪 Unit tests for velocity_metrics.py — ensures velocity magnitude calculations across varied inputs

import pytest
import math
from src.grid_modules.cell import Cell
from src.metrics.velocity_metrics import compute_max_velocity

def make_cell(vx, vy, vz):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=[vx, vy, vz], pressure=0.0, fluid_mask=True)

def test_empty_grid_returns_zero():
    assert compute_max_velocity([]) == 0.0

def test_single_cell_velocity_magnitude():
    cell = make_cell(3.0, 4.0, 0.0)  # mag = 5.0
    assert compute_max_velocity([cell]) == pytest.approx(5.0)

def test_multiple_cells_returns_max_magnitude():
    grid = [
        make_cell(1.0, 0.0, 0.0),
        make_cell(2.0, 2.0, 0.0),   # mag = ~2.83
        make_cell(3.0, 4.0, 0.0),   # mag = 5.0
        make_cell(5.0, 12.0, 0.0),  # mag = 13.0 (max)
    ]
    assert compute_max_velocity(grid) == pytest.approx(13.0)

def test_none_velocity_skipped():
    cell1 = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)
    cell2 = make_cell(0.0, 0.0, 2.0)  # mag = 2.0
    assert compute_max_velocity([cell1, cell2]) == pytest.approx(2.0)

def test_invalid_velocity_shape_skipped():
    bad = Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True)
    good = make_cell(0.0, 0.0, 3.0)  # mag = 3.0
    assert compute_max_velocity([bad, good]) == pytest.approx(3.0)

def test_negative_components_handled_correctly():
    cell = make_cell(-6.0, -8.0, 0.0)  # mag = 10.0
    assert compute_max_velocity([cell]) == pytest.approx(10.0)

def test_precision_cutoff():
    cell1 = make_cell(1.0, 1.0, 1.0)  # mag ≈ 1.732
    cell2 = make_cell(1.0, 1.0, 1.001)  # slightly higher
    result = compute_max_velocity([cell1, cell2])
    expected = round(math.sqrt(1.0**2 + 1.0**2 + 1.001**2), 5)
    assert result == expected

def test_all_invalid_velocity_returns_zero():
    grid = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True),
        Cell(x=0.0, y=0.0, z=0.0, velocity="bad", pressure=0.0, fluid_mask=True),
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True)
    ]
    assert compute_max_velocity(grid) == 0.0