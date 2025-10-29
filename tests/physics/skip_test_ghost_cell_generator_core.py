import pytest
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def test_ghost_cell_created_for_x_min_face():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "nx": 2, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [{
            "apply_faces": ["x_min"],
            "apply_to": ["velocity", "pressure"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 5.0,
            "type": "dirichlet"
        }]
    }
    fluid = make_cell(0.25, 0.5, 0.5)
    padded_grid, ghost_registry = generate_ghost_cells([fluid], config, debug=False)
    assert len(padded_grid) == 2
    assert len(ghost_registry) == 1
    ghost = list(ghost_registry.values())[0]
    assert ghost["face"] == "x_min"
    assert ghost["coordinate"] == (fluid.x - 0.5, fluid.y, fluid.z)

def test_ghost_cell_created_for_x_max_face():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "nx": 2, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["x_max"],
            "face_types": {"x_max": "inlet"},
            "default_type": "wall"
        },
        "boundary_conditions": [{
            "apply_faces": ["x_max"],
            "apply_to": ["velocity", "pressure"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 5.0,
            "type": "dirichlet"
        }]
    }
    fluid = make_cell(0.75, 0.5, 0.5)
    padded_grid, ghost_registry = generate_ghost_cells([fluid], config, debug=False)
    assert len(padded_grid) == 2
    assert len(ghost_registry) == 1
    ghost = list(ghost_registry.values())[0]
    assert ghost["face"] == "x_max"
    assert ghost["coordinate"] == (fluid.x + 0.5, fluid.y, fluid.z)

def test_ghost_cells_exclude_nonfluid_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [{
            "apply_faces": ["x_min"],
            "apply_to": ["velocity"],
            "velocity": [0.0, 0.0, 0.0],
            "type": "dirichlet"
        }]
    }
    solid = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    padded_grid, ghost_registry = generate_ghost_cells([solid], config, debug=False)
    assert len(padded_grid) == 1
    assert len(ghost_registry) == 0



