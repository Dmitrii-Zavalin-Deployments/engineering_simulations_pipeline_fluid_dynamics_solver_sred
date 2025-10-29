# tests/grid_modules/test_cell.py
# ✅ Validation suite for src/grid_modules/cell.py

from src.grid_modules.cell import Cell

def test_cell_instantiation_with_valid_data():
    cell = Cell(
        x=1.0,
        y=2.0,
        z=3.0,
        velocity=[0.1, 0.2, 0.3],
        pressure=101325.0,
        fluid_mask=True
    )
    assert cell.x == 1.0
    assert cell.y == 2.0
    assert cell.z == 3.0
    assert cell.velocity == [0.1, 0.2, 0.3]
    assert cell.pressure == 101325.0
    assert cell.fluid_mask is True

def test_cell_allows_zero_velocity_and_pressure():
    cell = Cell(
        x=0.0,
        y=0.0,
        z=0.0,
        velocity=[0.0, 0.0, 0.0],
        pressure=0.0,
        fluid_mask=True
    )
    assert cell.velocity == [0.0, 0.0, 0.0]
    assert cell.pressure == 0.0

def test_cell_can_be_solid():
    cell = Cell(
        x=5.0,
        y=5.0,
        z=5.0,
        velocity=[0.0, 0.0, 0.0],
        pressure=0.0,
        fluid_mask=False
    )
    assert cell.fluid_mask is False

def test_cell_supports_negative_coordinates():
    cell = Cell(
        x=-1.0,
        y=-2.0,
        z=-3.0,
        velocity=[1.0, 1.0, 1.0],
        pressure=100000.0,
        fluid_mask=True
    )
    assert cell.x == -1.0
    assert cell.y == -2.0
    assert cell.z == -3.0

def test_cell_velocity_vector_length():
    cell = Cell(
        x=0.0,
        y=0.0,
        z=0.0,
        velocity=[1.0, 2.0, 3.0],
        pressure=101000.0,
        fluid_mask=True
    )
    assert isinstance(cell.velocity, list)
    assert len(cell.velocity) == 3

def test_cell_repr_and_equality():
    cell1 = Cell(1.0, 2.0, 3.0, [0.1, 0.2, 0.3], 101325.0, True)
    cell2 = Cell(1.0, 2.0, 3.0, [0.1, 0.2, 0.3], 101325.0, True)
    assert cell1 == cell2
    assert repr(cell1).startswith("Cell(")



