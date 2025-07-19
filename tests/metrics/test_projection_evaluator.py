# tests/metrics/test_projection_evaluator.py
# 🧪 Unit tests for src/metrics/projection_evaluator.py

from src.grid_modules.cell import Cell
from src.metrics.projection_evaluator import calculate_projection_passes

def make_cell(v):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=v, pressure=0.0, fluid_mask=True)

def test_returns_minimum_passes_for_empty_grid():
    assert calculate_projection_passes([]) == 1

def test_returns_minimum_passes_for_no_valid_vectors():
    grid = [make_cell(None), make_cell([1.0]), make_cell("bad")]
    assert calculate_projection_passes(grid) == 1

def test_uniform_velocity_returns_one_pass():
    grid = [make_cell([2.0, 0.0, 0.0]), make_cell([2.0, 0.0, 0.0]), make_cell([2.0, 0.0, 0.0])]
    assert calculate_projection_passes(grid) == 1  # No variation

def test_small_variation_triggers_extra_pass():
    # magnitudes: 2.0, 2.0, 2.6 → avg = 2.2, max = 2.6, variation = 0.4 → int(variation // 0.5) = 0
    grid = [make_cell([2.0, 0.0, 0.0]), make_cell([2.0, 0.0, 0.0]), make_cell([2.0, 0.0, 1.0])]
    assert calculate_projection_passes(grid) == 1

def test_variation_exceeds_half_unit():
    # magnitudes: 2.0, 2.0, 3.0 → avg = 2.333, max = 3.0, variation ≈ 0.667 → 1 extra pass
    grid = [make_cell([2.0, 0.0, 0.0]), make_cell([2.0, 0.0, 0.0]), make_cell([3.0, 0.0, 0.0])]
    assert calculate_projection_passes(grid) == 2

def test_multiple_passes_for_high_variation():
    # magnitudes: 1.0, 1.0, 5.0 → avg = 2.333, max = 5.0, variation ≈ 2.667 → int(2.667 // 0.5) = 5
    grid = [make_cell([1.0, 0.0, 0.0]), make_cell([1.0, 0.0, 0.0]), make_cell([5.0, 0.0, 0.0])]
    assert calculate_projection_passes(grid) == 6



