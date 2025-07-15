# tests/test_divergence_metrics.py
# 🧪 Unit tests for compute_max_divergence — validates central difference logic and grid exclusions

import pytest
from src.grid_modules.cell import Cell
from src.metrics.divergence_metrics import compute_max_divergence

def make_cell(x, y, z, velocity, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid_mask)

def test_zero_grid_returns_zero():
    assert compute_max_divergence([], domain={}) == 0.0
    assert compute_max_divergence([], domain={"nx": 1, "ny": 1, "nz": 1}) == 0.0

def test_domain_missing_returns_zero():
    grid = [make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])]
    assert compute_max_divergence(grid, {}) == 0.0

def test_divergence_skips_solid_cells():
    grid = [
        make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid_mask=False),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0], fluid_mask=True),
        make_cell(-1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid_mask=True)
    ]
    domain = {"min_x": -1.0, "max_x": 1.0, "nx": 2,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    result = compute_max_divergence(grid, domain)
    assert isinstance(result, float)
    assert result > 0.0

def test_single_fluid_cell_without_neighbors_returns_zero():
    grid = [make_cell(0.0, 0.0, 0.0, [1.0, 1.0, 1.0])]
    domain = {"min_x": 0.0, "max_x": 1.0, "nx": 1,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence(grid, domain) == 0.0

def test_valid_neighbors_contribute_to_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0]),
        make_cell(-1.0, 0.0, 0.0, [-1.0, 0.0, 0.0])
    ]
    domain = {"min_x": -1.0, "max_x": 1.0, "nx": 2,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    result = compute_max_divergence(grid, domain)
    assert result == pytest.approx(1.0)  # (1 - -1)/(2*dx) = 1.0

def test_velocity_with_invalid_format_skipped_safely():
    grid = [
        make_cell(0.0, 0.0, 0.0, None),
        make_cell(1.0, 0.0, 0.0, [1.0, 0.0]),  # invalid length
        make_cell(-1.0, 0.0, 0.0, "not_a_vector")  # invalid type
    ]
    domain = {"min_x": -1.0, "max_x": 1.0, "nx": 2,
              "min_y": 0.0, "max_y": 1.0, "ny": 1,
              "min_z": 0.0, "max_z": 1.0, "nz": 1}
    assert compute_max_divergence(grid, domain) == 0.0

def test_multi_axis_contribution_to_divergence():
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0]),
        make_cell(-1.0, 0.0, 0.0, [-2.0, 0.0, 0.0]),
        make_cell(0.0, 1.0, 0.0, [0.0, 2.0, 0.0]),
        make_cell(0.0, -1.0, 0.0, [0.0, -2.0, 0.0]),
        make_cell(0.0, 0.0, 1.0, [0.0, 0.0, 2.0]),
        make_cell(0.0, 0.0, -1.0, [0.0, 0.0, -2.0])
    ]
    domain = {"min_x": -1.0, "max_x": 1.0, "nx": 2,
              "min_y": -1.0, "max_y": 1.0, "ny": 2,
              "min_z": -1.0, "max_z": 1.0, "nz": 2}
    result = compute_max_divergence(grid, domain)
    assert result == pytest.approx(3.0)