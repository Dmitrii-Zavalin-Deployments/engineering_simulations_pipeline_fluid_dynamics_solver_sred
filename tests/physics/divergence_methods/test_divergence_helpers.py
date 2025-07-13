# tests/physics/divergence_methods/test_divergence_helpers.py
# 🧪 Unit tests for divergence_helpers — neighbor lookup and gradient logic

import pytest
from src.physics.divergence_methods.divergence_helpers import (
    get_neighbor_velocity,
    central_gradient
)
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=fluid_mask)

# ----------------------------
# get_neighbor_velocity tests
# ----------------------------

def test_get_neighbor_velocity_returns_velocity_for_valid_fluid_neighbor():
    center = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    neighbor = make_cell(2.0, 0.0, 0.0, [2.0, 0.0, 0.0])
    lookup = {(center.x, center.y, center.z): center,
              (neighbor.x, neighbor.y, neighbor.z): neighbor}
    result = get_neighbor_velocity(lookup, center.x, center.y, center.z, 'x', +1, spacing=1.0)
    assert result == [2.0, 0.0, 0.0]

def test_get_neighbor_velocity_returns_none_if_neighbor_missing():
    center = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    lookup = {(center.x, center.y, center.z): center}
    result = get_neighbor_velocity(lookup, center.x, center.y, center.z, 'x', +1, spacing=1.0)
    assert result is None

def test_get_neighbor_velocity_returns_none_for_solid_neighbor():
    center = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    solid = make_cell(1.0, 0.0, 0.0, [2.0, 0.0, 0.0], fluid_mask=False)
    lookup = {(center.x, center.y, center.z): center,
              (solid.x, solid.y, solid.z): solid}
    result = get_neighbor_velocity(lookup, center.x, center.y, center.z, 'x', +1, spacing=1.0)
    assert result is None

def test_get_neighbor_velocity_returns_none_for_malformed_velocity():
    center = make_cell(1.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    bad = make_cell(2.0, 0.0, 0.0, "invalid")
    lookup = {(center.x, center.y, center.z): center,
              (bad.x, bad.y, bad.z): bad}
    result = get_neighbor_velocity(lookup, center.x, center.y, center.z, 'x', +1, spacing=1.0)
    assert result is None

# ----------------------------
# central_gradient tests
# ----------------------------

def test_central_gradient_returns_zero_if_any_neighbor_missing():
    assert central_gradient([2.0, 0.0, 0.0], None, 1.0, 0) == 0.0
    assert central_gradient(None, [1.0, 0.0, 0.0], 1.0, 0) == 0.0

def test_central_gradient_computes_correct_value():
    v_pos = [3.0, 2.0, 1.0]
    v_neg = [1.0, 0.0, -1.0]
    spacing = 1.0
    result = central_gradient(v_pos, v_neg, spacing, component=0)
    assert result == pytest.approx(1.0)

def test_central_gradient_handles_non_x_component():
    v_pos = [3.0, 5.0, 0.0]
    v_neg = [3.0, 1.0, 0.0]
    result_y = central_gradient(v_pos, v_neg, spacing=1.0, component=1)
    assert result_y == pytest.approx(2.0)



