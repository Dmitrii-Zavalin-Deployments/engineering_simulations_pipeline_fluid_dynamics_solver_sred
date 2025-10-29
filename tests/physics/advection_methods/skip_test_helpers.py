# tests/physics/advection_methods/test_helpers.py
# ✅ Validation suite for src/physics/advection_methods/helpers.py

import pytest
from src.physics.advection_methods.helpers import copy_cell, vector_add, vector_scale
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

# 🔁 copy_cell tests
def test_copy_cell_preserves_all_fields_by_default():
    original = make_cell(1.0, 2.0, 3.0, velocity=[1.0, 2.0, 3.0], pressure=5.0, fluid_mask=False)
    copied = copy_cell(original)
    assert copied.x == original.x
    assert copied.y == original.y
    assert copied.z == original.z
    assert copied.velocity == original.velocity
    assert copied.pressure == original.pressure
    assert copied.fluid_mask == original.fluid_mask
    assert copied is not original

def test_copy_cell_overrides_velocity_and_pressure():
    original = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 1.0, 1.0], pressure=1.0)
    new_velocity = [9.0, 8.0, 7.0]
    new_pressure = 42.0
    copied = copy_cell(original, velocity=new_velocity, pressure=new_pressure)
    assert copied.velocity == new_velocity
    assert copied.pressure == new_pressure
    assert copied.x == original.x
    assert copied.fluid_mask == original.fluid_mask

def test_copy_cell_handles_none_velocity_and_pressure():
    original = make_cell(0.0, 0.0, 0.0, velocity=[2.0, 2.0, 2.0], pressure=3.0)
    copied = copy_cell(original, velocity=None, pressure=None)
    assert copied.velocity == original.velocity
    assert copied.pressure == original.pressure

# ➕ vector_add tests
def test_vector_add_returns_correct_sum():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    result = vector_add(a, b)
    assert result == [5.0, 7.0, 9.0]

def test_vector_add_with_negative_values():
    a = [-1.0, -2.0, -3.0]
    b = [3.0, 2.0, 1.0]
    result = vector_add(a, b)
    assert result == [2.0, 0.0, -2.0]

# ✖️ vector_scale tests
def test_vector_scale_applies_scalar_correctly():
    v = [1.0, 2.0, 3.0]
    result = vector_scale(v, 2.0)
    assert result == [2.0, 4.0, 6.0]

def test_vector_scale_with_zero_scalar():
    v = [1.0, -1.0, 5.0]
    result = vector_scale(v, 0.0)
    assert result == [0.0, 0.0, 0.0]

def test_vector_scale_with_negative_scalar():
    v = [1.0, 2.0, 3.0]
    result = vector_scale(v, -1.0)
    assert result == [-1.0, -2.0, -3.0]



