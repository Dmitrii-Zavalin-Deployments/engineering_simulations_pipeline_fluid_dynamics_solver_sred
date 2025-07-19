# tests/metrics/test_cfl_controller.py
# 🧪 Unit tests for src/metrics/cfl_controller.py

import math
from src.grid_modules.cell import Cell
from src.metrics.cfl_controller import compute_global_cfl

def make_cell(velocity):
    return Cell(x=0.0, y=0.0, z=0.0, velocity=velocity, pressure=0.0, fluid_mask=True)

def test_compute_cfl_standard_case():
    grid = [make_cell([1.0, 0.0, 0.0]), make_cell([0.0, 2.0, 0.0])]
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 1}
    timestep = 0.1
    result = compute_global_cfl(grid, timestep, domain)
    expected = round(2.0 * timestep / (1.0 / 1), 5)
    assert result == expected

def test_compute_cfl_missing_domain_key_returns_zero():
    grid = [make_cell([1.0, 0.0, 0.0])]
    domain = {"min_x": 0.0, "max_x": 1.0}  # missing nx
    result = compute_global_cfl(grid, 0.1, domain)
    assert result == 0.0

def test_compute_cfl_empty_grid_returns_zero():
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 1}
    result = compute_global_cfl([], 0.1, domain)
    assert result == 0.0

def test_compute_cfl_handles_non_vector_velocity():
    grid = [make_cell(None), make_cell([1.0, 2.0]), make_cell("bad")]
    domain = {"min_x": 0.0, "max_x": 2.0, "nx": 2}
    result = compute_global_cfl(grid, 0.1, domain)
    assert result == 0.0

def test_compute_cfl_rounding_behavior():
    v = [1.0, 1.0, 1.0]  # magnitude ≈ 1.73205
    grid = [make_cell(v)]
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 2}
    timestep = 0.05
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    expected = round(math.sqrt(3) * timestep / dx, 5)
    result = compute_global_cfl(grid, timestep, domain)
    assert result == expected


