# tests/metrics/test_velocity_metrics.py
# 🧪 Unit tests for src/metrics/velocity_metrics.py

from src.grid_modules.cell import Cell
from src.metrics.velocity_metrics import compute_max_velocity

def make_cell(velocity):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=velocity, pressure=0.0, fluid_mask=True)

def test_returns_zero_for_empty_grid():
    assert compute_max_velocity([]) == 0.0

def test_returns_zero_for_non_vector_velocities():
    grid = [make_cell(None), make_cell("fast"), make_cell([1.0]), make_cell([1.0, 2.0])]
    assert compute_max_velocity(grid) == 0.0

def test_computes_max_velocity_for_uniform_vector():
    grid = [make_cell([3.0, 4.0, 0.0]), make_cell([3.0, 4.0, 0.0])]
    # Magnitude = sqrt(9 + 16) = 5.0
    assert compute_max_velocity(grid) == 5.0

def test_computes_max_velocity_for_mixed_vectors():
    grid = [
        make_cell([1.0, 1.0, 1.0]),     # ≈ 1.732
        make_cell([3.0, 4.0, 0.0]),     # = 5.0
        make_cell([6.0, 8.0, 0.0]),     # = 10.0
        make_cell([0.0, 0.0, 0.0])      # = 0.0
    ]
    assert compute_max_velocity(grid) == 10.0

def test_returns_rounded_velocity():
    grid = [make_cell([1.123456, 2.654321, 3.987654])]
    magnitude = compute_max_velocity(grid)
    assert isinstance(magnitude, float)
    assert round(magnitude, 5) == magnitude



