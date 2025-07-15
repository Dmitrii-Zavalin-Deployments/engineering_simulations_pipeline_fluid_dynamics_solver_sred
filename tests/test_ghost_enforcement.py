# tests/test_ghost_enforcement.py
# 🧪 Unit tests for ghost cell creation and boundary enforcement behavior

import pytest
from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.physics.boundary_condition_solver import apply_boundary_conditions

def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)

def make_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 2, "nz": 1
        },
        "boundary_conditions": {
            "apply_faces": ["x_min", "x_max"],
            "apply_to": ["pressure", "velocity"],
            "velocity": [0.4, 0.0, 0.0],
            "pressure": 20.0,
            "no_slip": False
        }
    }

def test_ghost_creation_and_face_tagging():
    config = make_config()
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    grid = [
        make_fluid_cell(0.5 * dx, 1.0, 0.5),               # near x_min
        make_fluid_cell(2.5 * dx, 1.0, 0.5)                # near x_max
    ]
    padded, registry = generate_ghost_cells(grid, config)

    ghost_cells = [c for c in padded if not c.fluid_mask]
    ghost_faces = [getattr(c, "ghost_face", None) for c in ghost_cells]

    assert len(ghost_cells) >= 2
    assert "x_min" in ghost_faces
    assert "x_max" in ghost_faces

def test_ghost_field_enforcement():
    config = make_config()
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    grid = [
        make_fluid_cell(0.5 * dx, 0.5, 0.5),               # near x_min
        make_fluid_cell(2.5 * dx, 1.5, 0.5)                # near x_max
    ]
    padded, registry = generate_ghost_cells(grid, config)
    updated = apply_boundary_conditions(padded, registry, config)

    ghost_cells = [c for c in updated if not c.fluid_mask]
    for ghost in ghost_cells:
        assert ghost.velocity == [0.4, 0.0, 0.0]
        assert ghost.pressure == 20.0

def test_fluid_cells_receive_boundary_velocity():
    config = make_config()
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    grid = [
        make_fluid_cell(0.5 * dx, 0.5, 0.5),
        make_fluid_cell(2.5 * dx, 1.5, 0.5)
    ]
    padded, registry = generate_ghost_cells(grid, config)
    updated = apply_boundary_conditions(padded, registry, config)

    fluid_cells = [c for c in updated if c.fluid_mask]
    for fluid in fluid_cells:
        assert fluid.velocity == [0.4, 0.0, 0.0]
        assert fluid.pressure == 20.0

def test_ghost_registry_metadata_integrity():
    config = make_config()
    dx = (config["domain_definition"]["max_x"] - config["domain_definition"]["min_x"]) / config["domain_definition"]["nx"]
    grid = [make_fluid_cell(0.5 * dx, 0.5, 0.5)]
    _, registry = generate_ghost_cells(grid, config)

    for meta in registry.values():
        assert meta["face"] in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        assert isinstance(meta["coordinate"], tuple)
        assert meta["velocity"] == [0.4, 0.0, 0.0]
        assert meta["pressure"] == 20.0



