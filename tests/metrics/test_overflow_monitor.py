# tests/test_overflow_monitor.py
# 🧪 Unit tests for overflow_monitor.py — validates velocity threshold detection for unstable flows

import pytest
import math
from src.grid_modules.cell import Cell
from src.metrics.overflow_monitor import detect_overflow

def make_cell(vx, vy, vz):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=[vx, vy, vz], pressure=0.0, fluid_mask=True)

def test_detects_exact_threshold_does_not_trigger():
    magnitude = 10.0
    # Choose velocity vector with magnitude exactly at the threshold
    vx = magnitude / math.sqrt(3)
    cell = make_cell(vx, vx, vx)
    assert detect_overflow([cell]) is False

def test_detects_above_threshold_triggers():
    cell = make_cell(10.1, 0.0, 0.0)
    assert detect_overflow([cell]) is True

def test_detects_below_threshold_safe():
    cell = make_cell(5.0, 5.0, 0.0)  # magnitude ≈ 7.07
    assert detect_overflow([cell]) is False

def test_empty_grid_returns_false():
    assert detect_overflow([]) is False

def test_none_velocity_skipped():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=0.0, fluid_mask=True)
    assert detect_overflow([cell]) is False

def test_malformed_velocity_skipped():
    bad_cells = [
        Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 2.0], pressure=0.0, fluid_mask=True),
        Cell(x=0.0, y=0.0, z=0.0, velocity="not_a_vector", pressure=0.0, fluid_mask=True),
    ]
    assert detect_overflow(bad_cells) is False

def test_multiple_cells_only_one_triggers():
    cells = [
        make_cell(2.0, 2.0, 2.0),
        make_cell(1.0, 1.0, 1.0),
        make_cell(11.0, 0.0, 0.0)  # this should trigger
    ]
    assert detect_overflow(cells) is True

def test_negative_velocity_components_trigger():
    cell = make_cell(-11.0, 0.0, 0.0)
    assert detect_overflow([cell]) is True

def test_velocity_on_threshold_borderline_float_precision():
    vx = 10.0 + 1e-6
    cell = make_cell(vx, 0.0, 0.0)
    assert detect_overflow([cell]) is True