# tests/physics/divergence_methods/test_divergence_helpers.py
# 🧪 Unit tests for src/physics/divergence_methods/divergence_helpers.py

from src.grid_modules.cell import Cell
from src.physics.divergence_methods.divergence_helpers import (
    get_neighbor_velocity,
    central_gradient
)

def make_cell(x, y, z, velocity=None, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=0.0, fluid_mask=fluid)

def test_get_neighbor_velocity_returns_valid_vector():
    lookup = {
        (1.0, 0.0, 0.0): make_cell(1.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0])
    }
    v = get_neighbor_velocity(lookup, 0.0, 0.0, 0.0, 'x', +1, spacing=1.0)
    assert v == [2.0, 0.0, 0.0]

def test_get_neighbor_velocity_returns_none_for_non_fluid():
    cell = make_cell(1.0, 0.0, 0.0, velocity=[2.0, 0.0, 0.0], fluid=False)
    lookup = {(1.0, 0.0, 0.0): cell}
    v = get_neighbor_velocity(lookup, 0.0, 0.0, 0.0, 'x', +1, spacing=1.0)
    assert v is None

def test_get_neighbor_velocity_returns_none_for_missing_neighbor():
    lookup = {}
    v = get_neighbor_velocity(lookup, 0.0, 0.0, 0.0, 'y', +1, spacing=1.0)
    assert v is None

def test_get_neighbor_velocity_returns_none_for_non_vector():
    cell = make_cell(1.0, 0.0, 0.0, velocity=None)
    lookup = {(1.0, 0.0, 0.0): cell}
    v = get_neighbor_velocity(lookup, 0.0, 0.0, 0.0, 'x', +1, spacing=1.0)
    assert v is None

def test_central_gradient_computes_difference_correctly():
    v_pos = [3.0, 6.0, 9.0]
    v_neg = [1.0, 4.0, 7.0]
    grad = central_gradient(v_pos, v_neg, spacing=1.0, component=0)
    assert grad == 1.0  # (3 - 1) / 2

def test_central_gradient_returns_zero_if_v_pos_missing():
    grad = central_gradient(None, [1.0, 2.0, 3.0], spacing=1.0, component=1)
    assert grad == 0.0

def test_central_gradient_returns_zero_if_v_neg_missing():
    grad = central_gradient([4.0, 5.0, 6.0], None, spacing=1.0, component=2)
    assert grad == 0.0

def test_central_gradient_returns_zero_if_both_missing():
    grad = central_gradient(None, None, spacing=1.0, component=0)
    assert grad == 0.0



