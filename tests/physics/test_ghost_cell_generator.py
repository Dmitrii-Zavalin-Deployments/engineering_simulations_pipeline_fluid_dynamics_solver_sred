# tests/physics/test_ghost_cell_generator.py
# 🧪 Unit tests for src/physics/ghost_cell_generator.py

from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells

def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=2.0, fluid_mask=True)

def test_generates_ghost_on_x_min_face():
    grid = [make_fluid_cell(0.0, 0.5, 0.5)]
    config = {
        "domain_definition": {"min_x": 0.0, "max_x": 1.0, "nx": 1, "ny": 1, "nz": 1},
        "boundary_conditions": {"apply_faces": ["x_min"]}
    }
    padded, registry = generate_ghost_cells(grid, config)
    assert len(padded) == 2
    ghost = padded[1]
    assert ghost.x < grid[0].x
    assert registry[id(ghost)]["face"] == "x_min"

def test_enforces_no_slip_velocity():
    grid = [make_fluid_cell(0.0, 0.5, 0.5)]
    config = {
        "domain_definition": {"min_x": 0.0, "max_x": 1.0, "nx": 1, "ny": 1, "nz": 1},
        "boundary_conditions": {
            "apply_faces": ["x_min"],
            "no_slip": True,
            "velocity": [9.9, 9.9, 9.9]
        }
    }
    padded, registry = generate_ghost_cells(grid, config)
    ghost = padded[-1]
    assert ghost.velocity == [0.0, 0.0, 0.0]

def test_enforced_pressure_is_applied():
    grid = [make_fluid_cell(1.0, 1.0, 1.0)]
    config = {
        "domain_definition": {"min_z": 0.0, "max_z": 1.0, "nx": 1, "ny": 1, "nz": 1},
        "boundary_conditions": {
            "apply_faces": ["z_max"],
            "pressure": 7.7
        }
    }
    padded, registry = generate_ghost_cells(grid, config)
    ghost = padded[-1]
    assert ghost.pressure == 7.7
    assert registry[id(ghost)]["pressure"] == 7.7

def test_skips_non_fluid_cells():
    solid = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False)
    config = {
        "domain_definition": {"min_y": 0.0, "max_y": 1.0, "nx": 1, "ny": 1, "nz": 1},
        "boundary_conditions": {
            "apply_faces": ["y_min"]
        }
    }
    padded, registry = generate_ghost_cells([solid], config)
    assert len(padded) == 1
    assert len(registry) == 0

def test_multiple_faces_generate_multiple_ghosts():
    grid = [make_fluid_cell(0.0, 0.0, 0.0)]
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "boundary_conditions": {
            "apply_faces": ["x_min", "y_min", "z_min"],
            "velocity": [1.0, 2.0, 3.0],
            "pressure": 4.0
        }
    }
    padded, registry = generate_ghost_cells(grid, config)
    assert len(padded) == 4  # 1 fluid + 3 ghost
    assert len(registry) == 3
    faces = set([meta["face"] for meta in registry.values()])
    assert faces == {"x_min", "y_min", "z_min"}

def test_registry_contains_expected_metadata():
    grid = [make_fluid_cell(0.0, 0.0, 1.0)]
    config = {
        "domain_definition": {
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "boundary_conditions": {
            "apply_faces": ["z_max"],
            "velocity": [2.2, 3.3, 4.4],
            "pressure": 5.5
        }
    }
    padded, registry = generate_ghost_cells(grid, config)
    ghost = padded[-1]
    meta = registry[id(ghost)]
    assert isinstance(meta["origin"], tuple)
    assert meta["velocity"] == [2.2, 3.3, 4.4]
    assert meta["pressure"] == 5.5



