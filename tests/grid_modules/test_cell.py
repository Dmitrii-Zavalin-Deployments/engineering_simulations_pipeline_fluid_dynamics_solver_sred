# tests/grid_modules/test_cell.py
# 🧪 Unit tests for src/grid_modules/cell.py

from src.grid_modules.cell import Cell

def test_cell_initialization_with_valid_values():
    c = Cell(x=1.0, y=2.0, z=3.0, velocity=[0.1, 0.2, 0.3], pressure=100.0, fluid_mask=True)
    assert c.x == 1.0
    assert c.y == 2.0
    assert c.z == 3.0
    assert c.velocity == [0.1, 0.2, 0.3]
    assert c.pressure == 100.0
    assert c.fluid_mask is True

def test_cell_all_zero_values():
    c = Cell(x=0.0, y=0.0, z=0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=False)
    assert c.x == 0.0
    assert c.y == 0.0
    assert c.z == 0.0
    assert c.velocity == [0.0, 0.0, 0.0]
    assert c.pressure == 0.0
    assert c.fluid_mask is False

def test_cell_negative_coordinates_and_pressure():
    c = Cell(x=-1.0, y=-2.5, z=-3.0, velocity=[-0.1, -0.2, -0.3], pressure=-50.0, fluid_mask=True)
    assert c.x == -1.0
    assert c.y == -2.5
    assert c.z == -3.0
    assert c.velocity == [-0.1, -0.2, -0.3]
    assert c.pressure == -50.0
    assert c.fluid_mask is True

def test_cell_velocity_vector_length():
    c = Cell(x=0, y=0, z=0, velocity=[0.5, -0.5, 1.0], pressure=10.0, fluid_mask=True)
    assert len(c.velocity) == 3



