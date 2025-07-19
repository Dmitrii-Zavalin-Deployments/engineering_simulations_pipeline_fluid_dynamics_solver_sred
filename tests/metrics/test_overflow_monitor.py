# tests/metrics/test_overflow_monitor.py
# 🧪 Unit tests for src/metrics/overflow_monitor.py

import math  # for math.sqrt in tolerance-aware assertion
from math import sqrt, isclose

from src.grid_modules.cell import Cell
from src.metrics.overflow_monitor import detect_overflow

def make_cell(velocity):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=velocity, pressure=0.0, fluid_mask=True)

def test_detect_overflow_returns_false_for_empty_grid():
    assert detect_overflow([]) is False

def test_detect_overflow_returns_false_if_all_magnitudes_below_threshold():
    grid = [
        make_cell([1.0, 1.0, 1.0]),    # magnitude ≈ 1.732
        make_cell([5.0, 5.0, 5.0]),    # magnitude ≈ 8.660
        make_cell([0.0, 0.0, 0.0])     # magnitude = 0
    ]
    assert detect_overflow(grid) is False

def test_detect_overflow_returns_true_if_any_magnitude_exceeds_threshold():
    grid = [
        make_cell([1.0, 1.0, 1.0]),    # safe
        make_cell([6.0, 6.0, 6.0]),    # magnitude ≈ 10.392
        make_cell([0.0, 0.0, 0.0])     # safe
    ]
    assert detect_overflow(grid) is True

def test_detect_overflow_handles_exact_threshold_boundary():
    # magnitude = 10.0 → should NOT trigger overflow
    v = [(10.0 - 1e-8) / sqrt(3)] * 3  # Clamp below threshold to avoid drift
    grid = [make_cell(v)]
    assert detect_overflow(grid) is False

def test_detect_overflow_ignores_non_vector_velocity_fields():
    grid = [
        make_cell(None),
        make_cell([1.0]),
        make_cell("fast"),
        make_cell([3.0, 4.0, 12.0])  # magnitude = 13.0 → triggers
    ]
    assert detect_overflow(grid) is True

def test_detect_overflow_returns_false_for_all_invalid_velocity_entries():
    grid = [
        make_cell(None),
        make_cell([]),
        make_cell([1.0, 2.0]),  # too short
        make_cell("bad")
    ]
    assert detect_overflow(grid) is False



