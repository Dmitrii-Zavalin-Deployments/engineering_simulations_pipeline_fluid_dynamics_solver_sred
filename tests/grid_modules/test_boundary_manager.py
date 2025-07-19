# tests/grid_modules/test_boundary_manager.py
# 🧪 Unit tests for src/grid_modules/boundary_manager.py

from src.grid_modules.boundary_manager import apply_boundaries
from src.grid_modules.cell import Cell

def make_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=(0, 0, 0), pressure=0.0, fluid_mask=True)

def test_boundary_tagging_applies_wall_and_interior():
    domain = {"nx": 3, "ny": 3, "nz": 3}
    cells = [
        make_cell(0, 0, 0),  # corner: wall
        make_cell(1, 1, 1),  # center: interior
        make_cell(2, 2, 2),  # opposite corner: wall
        make_cell(1, 0, 1),  # y edge: wall
    ]

    updated = apply_boundaries(cells, domain)
    tags = [c.boundary_type for c in updated]
    assert tags == ["wall", "interior", "wall", "wall"]

def test_empty_grid_returns_empty_list():
    domain = {"nx": 0, "ny": 0, "nz": 0}
    result = apply_boundaries([], domain)
    assert result == []

def test_all_cells_tagged_wall_on_1x1x1_domain():
    domain = {"nx": 1, "ny": 1, "nz": 1}
    cells = [make_cell(0, 0, 0)]
    updated = apply_boundaries(cells, domain)
    assert updated[0].boundary_type == "wall"

def test_interior_cell_on_large_grid():
    domain = {"nx": 10, "ny": 10, "nz": 10}
    cell = make_cell(5, 5, 5)
    updated = apply_boundaries([cell], domain)
    assert updated[0].boundary_type == "interior"

def test_edge_cell_on_high_resolution_face():
    domain = {"nx": 20, "ny": 20, "nz": 20}
    edge_cells = [
        make_cell(0, 10, 10),
        make_cell(19, 10, 10),
        make_cell(10, 0, 10),
        make_cell(10, 19, 10),
        make_cell(10, 10, 0),
        make_cell(10, 10, 19),
    ]
    updated = apply_boundaries(edge_cells, domain)
    for cell in updated:
        assert cell.boundary_type == "wall"



