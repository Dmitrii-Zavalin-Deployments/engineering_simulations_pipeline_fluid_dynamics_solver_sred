import pytest
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def test_ghost_registry_contains_enforcement_metadata():
    config = {
        "domain_definition": {
            "min_y": 0.0, "max_y": 1.0,
            "nx": 1, "ny": 2, "nz": 1
        },
        "ghost_rules": {
            "boundary_faces": ["y_max"],
            "face_types": {"y_max": "outlet"},
            "default_type": "wall"
        },
        "boundary_conditions": [{
            "apply_faces": ["y_max"],
            "apply_to": ["pressure"],
            "pressure": 10.0,
            "type": "dirichlet"
        }]
    }
    fluid = make_cell(0.5, 0.75, 0.5)
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
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 2
        },
        "ghost_rules": {
            "boundary_faces": ["z_min"],
            "face_types": {"z_min": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [{
            "apply_faces": ["z_min"],
            "apply_to": ["velocity"],
            "type": "dirichlet"
            # velocity missing
        }]
    }
    fluid = make_cell(0.5, 0.5, 0.25)
    with pytest.raises(ValueError, match="Missing velocity for face 'z_min'"):
        generate_ghost_cells([fluid], config, debug=False)

def test_raises_error_for_missing_dirichlet_pressure():
    config = {
        "domain_definition": {
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 2
        },
        "ghost_rules": {
            "boundary_faces": ["z_max"],
            "face_types": {"z_max": "wall"},
            "default_type": "wall"
        },
        "boundary_conditions": [{
            "apply_faces": ["z_max"],
            "apply_to": ["pressure"],
            "type": "dirichlet"
            # pressure missing
        }]
    }
    fluid = make_cell(0.5, 0.5, 0.75)
    with pytest.raises(ValueError, match="Missing pressure for face 'z_max'"):
        generate_ghost_cells([fluid], config, debug=False)



