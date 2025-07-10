# tests/grid/test_cell.py

import pytest
from src.grid_modules.cell import Cell
from dataclasses import asdict

def test_cell_initialization():
    cell = Cell(
        x=1.0,
        y=2.0,
        z=3.0,
        velocity=[0.5, -0.2, 0.0],
        pressure=100.0
    )
    assert cell.x == 1.0
    assert cell.y == 2.0
    assert cell.z == 3.0
    assert cell.velocity == [0.5, -0.2, 0.0]
    assert cell.pressure == 100.0

def test_cell_velocity_structure():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=[0.1, 0.1, 0.1], pressure=10.0)
    assert isinstance(cell.velocity, list)
    assert len(cell.velocity) == 3
    assert all(isinstance(v, float) for v in cell.velocity)

def test_cell_pressure_type():
    cell = Cell(x=0.0, y=0.0, z=0.0, velocity=[1.0, 0.0, 0.0], pressure=42.5)
    assert isinstance(cell.pressure, float)

def test_cell_asdict_serialization():
    cell = Cell(x=1.1, y=2.2, z=3.3, velocity=[0.0, 0.0, 1.0], pressure=99.9)
    cell_dict = asdict(cell)
    assert isinstance(cell_dict, dict)
    assert set(cell_dict.keys()) == {"x", "y", "z", "velocity", "pressure"}
    assert cell_dict["velocity"] == [0.0, 0.0, 1.0]
    assert cell_dict["pressure"] == 99.9



