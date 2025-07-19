# tests/physics/advection_methods/test_helpers.py
# 🧪 Unit tests for src/physics/advection_methods/helpers.py

from src.grid_modules.cell import Cell
from src.physics.advection_methods.helpers import (
    copy_cell,
    vector_add,
    vector_scale,
    vector_magnitude,
    interpolate_velocity
)

def test_copy_cell_preserves_original_when_no_override():
    c = Cell(x=1, y=2, z=3, velocity=[1.0, 2.0, 3.0], pressure=5.0, fluid_mask=True)
    copied = copy_cell(c)
    assert copied.x == c.x
    assert copied.y == c.y
    assert copied.z == c.z
    assert copied.velocity == [1.0, 2.0, 3.0]
    assert copied.pressure == 5.0
    assert copied.fluid_mask is True

def test_copy_cell_with_velocity_override():
    c = Cell(x=0, y=0, z=0, velocity=[0, 0, 0], pressure=1.0, fluid_mask=False)
    v = [3.0, 2.0, 1.0]
    copied = copy_cell(c, velocity=v)
    assert copied.velocity == v
    assert copied.pressure == 1.0

def test_copy_cell_with_pressure_override():
    c = Cell(x=0, y=0, z=0, velocity=[0, 0, 0], pressure=1.0, fluid_mask=False)
    copied = copy_cell(c, pressure=9.5)
    assert copied.pressure == 9.5
    assert copied.velocity == [0, 0, 0]

def test_vector_add_returns_correct_sum():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert vector_add(a, b) == [5.0, 7.0, 9.0]

def test_vector_scale_applies_scalar_correctly():
    v = [1.0, -2.0, 0.5]
    s = 2.0
    assert vector_scale(v, s) == [2.0, -4.0, 1.0]

def test_vector_magnitude_computes_norm():
    v = [3.0, 4.0, 0.0]
    assert vector_magnitude(v) == 5.0

def test_interpolate_velocity_weight_zero():
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    result = interpolate_velocity(v1, v2, 0.0)
    assert result == v1

def test_interpolate_velocity_weight_one():
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    result = interpolate_velocity(v1, v2, 1.0)
    assert result == v2

def test_interpolate_velocity_half_weight():
    v1 = [1.0, 2.0, 3.0]
    v2 = [3.0, 4.0, 5.0]
    result = interpolate_velocity(v1, v2, 0.5)
    assert result == [2.0, 3.0, 4.0]



