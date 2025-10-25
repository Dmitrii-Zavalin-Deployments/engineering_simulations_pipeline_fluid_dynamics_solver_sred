# tests/physics/test_ghost_cell_generator.py
# ✅ Validation suite for src/physics/ghost_cell_generator.py

import pytest
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity,
        pressure=pressure,
        fluid_mask=fluid_mask
    )

def test_ghost_cell_created_for_x_min_face():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 1, "nz": 1  # dx = 0.5
        },
        "ghost_rules": {
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [
            {
                "apply_faces": ["x_min"],
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 5.0,
                "type": "dirichlet"
            }
        ]
    }

    fluid = make_cell(0.0, 0.5, 0.5)  # ✅ adjacent to x_min
    padded_grid, ghost_registry = generate_ghost_cells([fluid], config, debug=False)

    assert len(padded_grid) == 2  # original + 1 ghost
    assert len(ghost_registry) == 1

    ghost = list(ghost_registry.values())[0]
    assert ghost["face"] == "x_min"
    assert ghost["coordinate"] == (fluid.x - 0.5, fluid.y, fluid.z)

def test_ghost_cell_created_for_x_max_face():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 1, "nz": 1  # dx = 0.5
        },
        "ghost_rules": {
            "boundary_faces": ["x_max"],
            "face_types": {"x_max": "inlet"},
            "default_type": "wall"
        },
        "boundary_conditions": [
            {
                "apply_faces": ["x_max"],
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 5.0,
                "type": "dirichlet"
            }
        ]
    }

    fluid = make_cell(1.0, 0.5, 0.5)  # ✅ adjacent to x_max
    padded_grid, ghost_registry = generate_ghost_cells([fluid], config, debug=False)

    assert len(padded_grid) == 2  # original + 1 ghost
    assert len(ghost_registry) == 1

    ghost = list(ghost_registry.values())[0]
    assert ghost["face"] == "x_max"
    assert ghost["coordinate"] == (fluid.x + 0.5, fluid.y, fluid.z)

def test_ghost_cells_exclude_nonfluid_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [
            {
                "apply_faces": ["x_min"],
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "type": "dirichlet"
            }
        ]
    }

    solid = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    padded_grid, ghost_registry = generate_ghost_cells([solid], config, debug=False)

    assert len(padded_grid) == 1  # no ghost added
    assert len(ghost_registry) == 0

def test_ghost_registry_contains_enforcement_metadata():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["y_max"],
            "face_types": {"y_max": "outlet"},
            "default_type": "wall"
        },
        "boundary_conditions": [
            {
                "apply_faces": ["y_max"],
                "apply_to": ["pressure"],
                "pressure": 10.0,
                "type": "dirichlet"
            }
        ]
    }

    fluid = make_cell(0.5, 1.0, 0.5)
    padded_grid, ghost_registry = generate_ghost_cells([fluid], config, debug=False)

    assert len(ghost_registry) == 1
    meta = list(ghost_registry.values())[0]
    assert meta["face"] == "y_max"
    assert meta["origin"] == (fluid.x, fluid.y, fluid.z)
    assert meta["pressure"] == 10.0
    assert meta["enforcement"]["pressure"] is True
    assert meta["enforcement"]["velocity"] is False

def test_raises_error_for_missing_dirichlet_velocity():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["z_min"],
            "face_types": {"z_min": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [
            {
                "apply_faces": ["z_min"],
                "apply_to": ["velocity"],
                "type": "dirichlet"
                # velocity missing
            }
        ]
    }

    fluid = make_cell(0.5, 0.5, 0.0)
    with pytest.raises(ValueError, match="Missing velocity for face 'z_min'"):
        generate_ghost_cells([fluid], config, debug=False)

def test_raises_error_for_missing_dirichlet_pressure():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["z_max"],
            "face_types": {"z_max": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [
            {
                "apply_faces": ["z_max"],
                "apply_to": ["pressure"],
                "type": "dirichlet"
                # pressure missing
            }
        ]
    }

    fluid = make_cell(0.5, 0.5, 1.0)
    with pytest.raises(ValueError, match="Missing pressure for face 'z_max'"):
        generate_ghost_cells([fluid], config, debug=False)



