# tests/grid/test_cell.py
# 🧪 Unit tests for Cell dataclass — verifies structure, types, and behavioral expectations

import pytest
from src.grid_modules.cell import Cell

def test_cell_instantiation():
    cell = Cell(x=1.0, y=2.0, z=3.0, velocity=[0.1, 0.0, -0.1], pressure=100.0, fluid_mask=True)
    assert isinstance(cell, Cell)
    assert cell.x == 1.0
    assert cell.y == 2.0
    assert cell.z == 3.0
    assert cell.velocity == [0.1, 0.0, -0.1]
    assert cell.pressure == 100.0
    assert cell.fluid_mask is True

def test_cell_with_solid_mask_has_valid_fields():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False)
    assert cell.velocity is None
    assert cell.pressure is None
    assert cell.fluid_mask is False

def test_cell_velocity_is_vector_of_length_three():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    assert isinstance(cell.velocity, list)
    assert len(cell.velocity) == 3
    for component in cell.velocity:
        assert isinstance(component, (int, float))

def test_cell_pressure_is_numeric_for_fluid():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0], pressure=42.5, fluid_mask=True)
    assert isinstance(cell.pressure, (int, float))
    assert cell.pressure >= 0

def test_cell_edge_values():
    cell = Cell(x=-1e6, y=0.0, z=1e6, velocity=[1e6, -1e6, 0.0], pressure=1e6, fluid_mask=True)
    assert isinstance(cell.x, float)
    assert isinstance(cell.y, float)
    assert isinstance(cell.z, float)
    assert isinstance(cell.pressure, float)
    assert all(isinstance(v, float) for v in cell.velocity)

def test_cell_repr_and_equality():
    c1 = Cell(x=1.0, y=2.0, z=3.0, velocity=[1.0, 0.0, -1.0], pressure=75.0, fluid_mask=True)
    c2 = Cell(x=1.0, y=2.0, z=3.0, velocity=[1.0, 0.0, -1.0], pressure=75.0, fluid_mask=True)
    assert repr(c1) == repr(c2)
    assert c1 == c2

def test_invalid_velocity_length_downgrades_cell():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=[0.0, 0.0], pressure=0.0, fluid_mask=True)
    # Expect fallback downgrade: velocity is reset, cell becomes solid
    assert cell.velocity is None
    assert cell.fluid_mask is False
    assert cell.pressure is None

def test_invalid_velocity_type():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity="not-a-vector", pressure=0.0, fluid_mask=True)
    # Velocity type is incorrect; fallback should trigger
    assert cell.velocity is None
    assert cell.fluid_mask is False
    assert cell.pressure is None



