# tests/physics/advection_methods/test_helpers.py
# 🧪 Unit tests for advection_helpers — validates vector math, interpolation, and cell cloning

import pytest
import math
from src.grid_modules.cell import Cell
from src.physics.advection_methods.helpers import (
    copy_cell,
    vector_add,
    vector_scale,
    vector_magnitude,
    interpolate_velocity
)

def test_copy_cell_preserves_all_attributes():
    original = Cell(x=1.0, y=2.0, z=3.0, velocity=[1.0, 0.0, 0.0], pressure=42.0, fluid_mask=True)
    cloned = copy_cell(original)
    # ✅ Confirm object identity is distinct
    assert cloned is not original
    # ✅ Confirm content equivalence
    assert cloned == original
    # ✅ Confirm field values preserved
    assert cloned.x == 1.0
    assert cloned.y == 2.0
    assert cloned.z == 3.0
    assert cloned.velocity == [1.0, 0.0, 0.0]
    assert cloned.pressure == 42.0
    assert cloned.fluid_mask is True

def test_copy_cell_overrides_velocity_and_pressure():
    original = Cell(x=0.0, y=0.0, z=0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=False)
    updated = copy_cell(original, velocity=[2.0, 2.0, 2.0], pressure=99.9)
    assert updated.velocity == [2.0, 2.0, 2.0]
    assert updated.pressure == 99.9
    assert updated.fluid_mask is False

def test_vector_add_correctness():
    v1 = [1.0, 2.0, 3.0]
    v2 = [0.5, -1.0, 0.0]
    result = vector_add(v1, v2)
    assert result == [1.5, 1.0, 3.0]

def test_vector_scale_applies_scalar():
    v = [2.0, -3.0, 4.0]
    result = vector_scale(v, 0.5)
    assert result == [1.0, -1.5, 2.0]

def test_vector_scale_zero_returns_zero_vector():
    v = [7.0, -8.0, 9.0]
    result = vector_scale(v, 0.0)
    assert result == [0.0, 0.0, 0.0]

def test_vector_magnitude_3_4_5():
    v = [3.0, 4.0, 0.0]
    result = vector_magnitude(v)
    assert result == pytest.approx(5.0)

def test_vector_magnitude_negative_components():
    v = [-6.0, -8.0, 0.0]
    result = vector_magnitude(v)
    assert result == pytest.approx(10.0)

def test_interpolate_velocity_midpoint():
    v1 = [0.0, 0.0, 0.0]
    v2 = [2.0, 4.0, 6.0]
    result = interpolate_velocity(v1, v2, 0.5)
    assert result == [1.0, 2.0, 3.0]

def test_interpolate_velocity_weight_zero_returns_v1():
    v1 = [9.0, 8.0, 7.0]
    v2 = [0.0, 0.0, 0.0]
    result = interpolate_velocity(v1, v2, 0.0)
    assert result == v1

def test_interpolate_velocity_weight_one_returns_v2():
    v1 = [9.0, 8.0, 7.0]
    v2 = [0.0, 0.0, 0.0]
    result = interpolate_velocity(v1, v2, 1.0)
    assert result == v2

def test_interpolate_velocity_nontrivial_weight():
    v1 = [1.0, 1.0, 1.0]
    v2 = [3.0, 3.0, 3.0]
    result = interpolate_velocity(v1, v2, 0.25)
    assert result == [1.5, 1.5, 1.5]



