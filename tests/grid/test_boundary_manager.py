# tests/grid/test_boundary_manager.py

import pytest
from src.grid_modules.cell import Cell
from src.grid_modules.boundary_manager import apply_boundaries

def make_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=1.0)

# ✅ Test: Correct tagging of edge and interior cells
def test_boundary_tagging():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 3, "ny": 3, "nz": 3
    }

    grid = [
        make_cell(0, 0, 0),    # corner
        make_cell(1, 1, 1),    # interior
        make_cell(2, 2, 2),    # corner opposite
        make_cell(0, 1, 1),    # x face
        make_cell(1, 0, 1),    # y face
        make_cell(1, 1, 2)     # z face
    ]

    tagged = apply_boundaries(grid, domain)

    results = [cell.boundary_type for cell in tagged]
    assert results == [
        "wall", "interior", "wall",
        "wall", "wall", "wall"
    ]

# ❌ Test: Missing domain keys
def test_missing_domain_keys():
    domain = {"nx": 3, "ny": 3, "nz": 3}  # min/max missing
    grid = [make_cell(0, 0, 0)]
    with pytest.raises(KeyError):
        apply_boundaries(grid, domain)

# ❌ Test: Non-numeric domain resolution
def test_invalid_domain_resolution():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": "high", "ny": 3, "nz": 3
    }
    grid = [make_cell(0, 0, 0)]
    with pytest.raises(TypeError):
        apply_boundaries(grid, domain)

# ✅ Test: Empty grid returns unchanged
def test_empty_grid_handling():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 3, "ny": 3, "nz": 3
    }
    tagged = apply_boundaries([], domain)
    assert tagged == []

# ✅ Test: All edge cells tagged correctly on small grid
def test_all_edges_on_minimal_grid():
    domain = {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 2, "ny": 2, "nz": 2
    }
    grid = [make_cell(x, y, z) for x in range(2) for y in range(2) for z in range(2)]
    tagged = apply_boundaries(grid, domain)
    for cell in tagged:
        assert cell.boundary_type == "wall"

# ✅ Test: Interior cell remains untagged
def test_single_interior_cell():
    domain = {
        "min_x": 0.0, "max_x": 3.0,
        "min_y": 0.0, "max_y": 3.0,
        "min_z": 0.0, "max_z": 3.0,
        "nx": 4, "ny": 4, "nz": 4
    }
    grid = [make_cell(2, 2, 2)]
    tagged = apply_boundaries(grid, domain)
    assert tagged[0].boundary_type == "interior"



