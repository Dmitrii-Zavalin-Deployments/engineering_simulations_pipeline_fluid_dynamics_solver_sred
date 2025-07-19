# tests/physics/test_advection.py
# 🧪 Unit tests for src/physics/advection.py

from src.grid_modules.cell import Cell
from src.physics.advection import compute_advection

def make_cell(x, velocity, fluid=True):
    return Cell(x=x, y=0.0, z=0.0, velocity=velocity, pressure=0.0, fluid_mask=fluid)

def test_skips_ghost_cells_from_advection():
    cell1 = make_cell(0.0, [1.0, 0.0, 0.0])
    cell2 = make_cell(1.0, [2.0, 0.0, 0.0])
    registry = {id(cell2)}
    result = compute_advection([cell1, cell2], dt=0.1, config={}, ghost_registry=registry)
    assert len(result) == 1
    assert result[0].x == 0.0

def test_skips_ghost_with_symmetry_metadata():
    cell1 = make_cell(0.0, [1.0, 0.0, 0.0])
    cell2 = make_cell(1.0, [2.0, 0.0, 0.0])
    registry = {id(cell2)}
    metadata = {id(cell2): {"boundary_type": "symmetry"}}
    result = compute_advection([cell1, cell2], dt=0.1, config={}, ghost_registry=registry, ghost_metadata=metadata)
    assert len(result) == 1
    assert result[0].x == 0.0

def test_preserves_non_fluid_cell_velocity():
    solid = make_cell(0.0, [3.0, 2.0, 1.0], fluid=False)
    result = compute_advection([solid], dt=0.1, config={})
    assert result[0].velocity == [3.0, 2.0, 1.0]

def test_updates_fluid_cell_velocity_with_valid_neighbor():
    cell_a = make_cell(0.0, [0.0, 0.0, 0.0])
    cell_b = make_cell(0.1, [2.0, 0.0, 0.0])
    config = {
        "domain_definition": {
            "nx": 10, "min_x": 0.0, "max_x": 1.0,
            "ny": 1, "max_y": 1.0, "min_y": 0.0,
            "nz": 1, "max_z": 1.0, "min_z": 0.0
        }
    }
    result = compute_advection([cell_a, cell_b], dt=0.05, config=config)
    assert result[1].velocity != [2.0, 0.0, 0.0]

def test_returns_empty_when_all_ghosts():
    ghost1 = make_cell(0.0, [1.0, 0.0, 0.0])
    ghost2 = make_cell(1.0, [2.0, 0.0, 0.0])
    registry = {id(ghost1), id(ghost2)}
    result = compute_advection([ghost1, ghost2], dt=0.1, config={}, ghost_registry=registry)
    assert result == []



