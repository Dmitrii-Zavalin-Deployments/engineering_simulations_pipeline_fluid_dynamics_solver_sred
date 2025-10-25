# tests/physics/divergence_methods/test_divergence_helpers.py
# ✅ Validation suite for src/physics/divergence_methods/divergence_helpers.py

import pytest
from src.physics.divergence_methods.divergence_helpers import get_neighbor_velocity, central_difference
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

# 🧭 get_neighbor_velocity tests
def test_get_neighbor_velocity_returns_correct_vector():
    spacing = 1.0
    center = make_cell(0.0, 0.0, 0.0)
    neighbor = make_cell(1.0, 0.0, 0.0, velocity=[1.0, 2.0, 3.0])
    grid = {
        (center.x, center.y, center.z): center,
        (neighbor.x, neighbor.y, neighbor.z): neighbor
    }

    result = get_neighbor_velocity(grid, 0.0, 0.0, 0.0, axis='x', sign=+1, spacing=spacing)
    assert result == [1.0, 2.0, 3.0]

def test_get_neighbor_velocity_returns_none_for_non_fluid():
    spacing = 1.0
    solid = make_cell(1.0, 0.0, 0.0, velocity=[9.0, 9.0, 9.0], fluid_mask=False)
    grid = {(1.0, 0.0, 0.0): solid}
    result = get_neighbor_velocity(grid, 0.0, 0.0, 0.0, axis='x', sign=+1, spacing=spacing)
    assert result is None

def test_get_neighbor_velocity_returns_none_for_missing_neighbor():
    spacing = 1.0
    grid = {}
    result = get_neighbor_velocity(grid, 0.0, 0.0, 0.0, axis='y', sign=-1, spacing=spacing)
    assert result is None

def test_get_neighbor_velocity_returns_none_for_invalid_velocity():
    spacing = 1.0
    bad = make_cell(0.0, 1.0, 0.0, velocity=None)
    grid = {(0.0, 1.0, 0.0): bad}
    result = get_neighbor_velocity(grid, 0.0, 0.0, 0.0, axis='y', sign=+1, spacing=spacing)
    assert result is None

# ∇ central_difference tests
def test_central_difference_computes_gradient_correctly():
    v_pos = [3.0, 0.0, 0.0]
    v_neg = [1.0, 0.0, 0.0]
    spacing = 1.0
    result = central_difference(v_pos, v_neg, spacing, component=0)
    assert result == 1.0

def test_central_difference_returns_zero_if_missing_neighbors():
    spacing = 1.0
    assert central_difference(None, [1.0, 0.0, 0.0], spacing, 0) == 0.0
    assert central_difference([1.0, 0.0, 0.0], None, spacing, 0) == 0.0
    assert central_difference(None, None, spacing, 0) == 0.0

def test_central_difference_handles_nonzero_component_indices():
    v_pos = [0.0, 6.0, 0.0]
    v_neg = [0.0, 2.0, 0.0]
    spacing = 2.0
    result = central_difference(v_pos, v_neg, spacing, component=1)
    assert result == 1.0



