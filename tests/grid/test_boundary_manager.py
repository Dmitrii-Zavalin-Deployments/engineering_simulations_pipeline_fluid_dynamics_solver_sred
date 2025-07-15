# tests/test_boundary_manager.py
# 🧪 Unit tests for boundary_manager.py — tags edge cells using resolution-based logic

import pytest
from src.grid_modules.cell import Cell
from src.grid_modules.boundary_manager import apply_boundaries

def make_cell(ix, iy, iz):
    # Creates a test cell with integer coordinates
    return Cell(x=ix, y=iy, z=iz, velocity=None, pressure=None, fluid_mask=True)

@pytest.mark.parametrize("domain,cells,expected_types", [
    (
        {"nx": 3, "ny": 3, "nz": 3},
        [make_cell(0, 0, 0), make_cell(1, 1, 1), make_cell(2, 2, 2)],
        ["wall", "interior", "wall"]
    ),
    (
        {"nx": 1, "ny": 1, "nz": 1},
        [make_cell(0, 0, 0)],
        ["wall"]
    ),
    (
        {"nx": 0, "ny": 3, "nz": 3},
        [make_cell(0, 0, 0), make_cell(0, 2, 2)],
        ["wall", "wall"]
    ),
    (
        {"nx": 3, "ny": 0, "nz": 3},
        [make_cell(0, 0, 0), make_cell(2, 0, 2)],
        ["wall", "wall"]
    ),
    (
        {"nx": 3, "ny": 3, "nz": 0},
        [make_cell(0, 0, 0), make_cell(2, 2, 0)],
        ["wall", "wall"]
    ),
    (
        {"nx": 5, "ny": 5, "nz": 5},
        [make_cell(0, 2, 2), make_cell(4, 2, 2), make_cell(2, 0, 2), make_cell(2, 4, 2), make_cell(2, 2, 0), make_cell(2, 2, 4)],
        ["wall", "wall", "wall", "wall", "wall", "wall"]
    ),
    (
        {"nx": 5, "ny": 5, "nz": 5},
        [make_cell(2, 2, 2)],
        ["interior"]
    )
])
def test_apply_boundaries_behavior(domain, cells, expected_types):
    tagged = apply_boundaries(cells, domain)
    for c, expected in zip(tagged, expected_types):
        assert hasattr(c, "boundary_type"), "❌ Missing 'boundary_type' attribute"
        assert c.boundary_type == expected, f"❌ Expected {expected} but got {c.boundary_type} at ({c.x}, {c.y}, {c.z})"

def test_boundary_type_assignment_on_large_grid():
    domain = {"nx": 10, "ny": 10, "nz": 10}
    grid = [make_cell(x, y, z) for x in range(10) for y in range(10) for z in range(10)]
    tagged = apply_boundaries(grid, domain)
    for cell in tagged:
        if cell.x in {0, 9} or cell.y in {0, 9} or cell.z in {0, 9}:
            assert cell.boundary_type == "wall"
        else:
            assert cell.boundary_type == "interior"

def test_empty_grid_returns_empty():
    result = apply_boundaries([], {"nx": 5, "ny": 5, "nz": 5})
    assert isinstance(result, list)
    assert result == []

def test_zero_resolution_sets_all_to_wall():
    grid = [make_cell(0, 0, 0), make_cell(1, 1, 1)]
    tagged = apply_boundaries(grid, {"nx": 0, "ny": 0, "nz": 0})
    for cell in tagged:
        assert cell.boundary_type == "wall"