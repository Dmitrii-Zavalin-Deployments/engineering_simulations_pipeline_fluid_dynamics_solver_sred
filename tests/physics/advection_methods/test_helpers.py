# tests/physics/advection_methods/test_helpers.py
# 🧪 Unit tests for advection_helpers.py — vector math and cell copying

import pytest
from src.physics.advection_methods.helpers import (
    copy_cell,
    vector_add,
    vector_scale,
    vector_magnitude,
    interpolate_velocity
)
from src.grid_modules.cell import Cell

# -----------------------
# copy_cell() test cases
# -----------------------

def test_copy_cell_preserves_fields_by_default():
    original = Cell(x=1, y=2, z=3, velocity=[1.0, 0.0, 0.0], pressure=2.5, fluid_mask=True)
    result = copy_cell(original)
    assert isinstance(result, Cell)
    assert result.x == original.x
    assert result.y == original.y
    assert result.z == original.z
    assert result.velocity == original.velocity
    assert result.pressure == original.pressure
    assert result.fluid_mask == original.fluid_mask

def test_copy_cell_overrides_velocity_only():
    original = Cell(x=0, y=0, z=0, velocity=[1.0, 2.0, 3.0], pressure=1.0, fluid_mask=True)
    new_velocity = [9.0, 8.0, 7.0]
    result = copy_cell(original, velocity=new_velocity)
    assert result.velocity == new_velocity
    assert result.pressure == original.pressure

def test_copy_cell_overrides_pressure_only():
    original = Cell(x=0, y=0, z=0, velocity=[0.5, 0.5, 0.5], pressure=1.2, fluid_mask=True)
    result = copy_cell(original, pressure=9.9)
    assert result.velocity == original.velocity
    assert result.pressure == 9.9

def test_copy_cell_with_both_overrides():
    original = Cell(x=5, y=4, z=3, velocity=[1, 1, 1], pressure=5.0, fluid_mask=False)
    result = copy_cell(original, velocity=[0, 0, 0], pressure=0.0)
    assert result.velocity == [0, 0, 0]
    assert result.pressure == 0.0

# --------------------------
# vector_add() test cases
# --------------------------

def test_vector_add_basic():
    assert vector_add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]

def test_vector_add_negatives():
    assert vector_add([-1, -2, -3], [1, 2, 3]) == [0, 0, 0]

# ---------------------------
# vector_scale() test cases
# ---------------------------

def test_vector_scale_positive():
    assert vector_scale([1.0, 2.0, 3.0], 2.0) == [2.0, 4.0, 6.0]

def test_vector_scale_zero():
    assert vector_scale([1.0, -1.0, 0.5], 0.0) == [0.0, 0.0, 0.0]

def test_vector_scale_negative():
    assert vector_scale([1.0, 2.0, 3.0], -1.0) == [-1.0, -2.0, -3.0]

# ------------------------------
# vector_magnitude() test cases
# ------------------------------

def test_vector_magnitude_zero():
    assert vector_magnitude([0.0, 0.0, 0.0]) == 0.0

def test_vector_magnitude_unit():
    assert pytest.approx(vector_magnitude([1.0, 0.0, 0.0])) == 1.0
    assert pytest.approx(vector_magnitude([0.0, 3.0, 4.0])) == 5.0

# ------------------------------------
# interpolate_velocity() test cases
# ------------------------------------

def test_interpolate_velocity_halfway():
    v1 = [1.0, 0.0, 0.0]
    v2 = [3.0, 2.0, 0.0]
    result = interpolate_velocity(v1, v2, 0.5)
    assert result == [2.0, 1.0, 0.0]

def test_interpolate_velocity_weight_zero():
    v1 = [1.0, 2.0, 3.0]
    v2 = [9.0, 9.0, 9.0]
    result = interpolate_velocity(v1, v2, 0.0)
    assert result == v1

def test_interpolate_velocity_weight_one():
    v1 = [0.0, 0.0, 0.0]
    v2 = [2.0, 4.0, 6.0]
    result = interpolate_velocity(v1, v2, 1.0)
    assert result == v2



