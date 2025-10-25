# tests/initialization/test_fluid_mask_initializer.py
# ✅ Validation suite for src/initialization/fluid_mask_initializer.py

from src.initialization.fluid_mask_initializer import initialize_masks, build_simulation_grid
from src.grid_modules.cell import Cell

def mock_cell(x, y, z, velocity=None, pressure=None):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure or 0.0,
        fluid_mask=True
    )

def test_initialize_masks_applies_fluid_mask_and_reflex_metadata():
    grid = [
        mock_cell(0.0, 0.5, 0.5),  # xmin → inlet
        mock_cell(1.0, 0.5, 0.5),  # xmax → outlet
        mock_cell(0.5, 0.0, 0.5),  # ymin → wall
        mock_cell(0.5, 1.0, 0.5),  # ymax → wall
        mock_cell(0.5, 0.5, 0.0),  # zmin → wall
        mock_cell(0.5, 0.5, 1.0),  # zmax → wall
        mock_cell(0.5, 0.5, 0.5)   # interior → fluid
    ]

    ghost_rules = {
        "boundary_faces": ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"],
        "face_types": {
            "xmin": "inlet",
            "xmax": "outlet",
            "ymin": "wall",
            "ymax": "wall",
            "zmin": "wall",
            "zmax": "wall"
        },
        "default_type": "wall"
    }

    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "ghost_rules": ghost_rules,
        "step_index": 42
    }

    result = initialize_masks(grid, config)
    assert len(result) == 7

    fluid_cells = [c for c in result if c.fluid_mask]
    ghost_cells = [c for c in result if not c.fluid_mask]

    assert len(fluid_cells) == 1
    assert len(ghost_cells) == 6

    for cell in ghost_cells:
        assert cell.ghost_face in {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax"}
        assert cell.boundary_tag == cell.ghost_face
        assert cell.ghost_type in {"inlet", "outlet", "wall"}
        assert cell.originated_from_boundary is True
        assert cell.mutation_triggered_by == "boundary_enforcement"
        assert cell.ghost_source_step == 42
        assert isinstance(cell.was_enforced, bool)

def test_initialize_masks_defaults_to_wall_when_face_type_missing():
    grid = [mock_cell(0.0, 0.0, 0.0)]  # xmin, ymin, zmin
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "ghost_rules": {
            "face_types": {},  # no overrides
            "default_type": "wall",
            "boundary_faces": []
        }
    }

    result = initialize_masks(grid, config)
    assert result[0].fluid_mask is False
    assert result[0].ghost_type == "wall"
    assert result[0].was_enforced is False

def test_initialize_masks_handles_minimal_config_safely():
    grid = [mock_cell(0.0, 0.0, 0.0)]
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0
        },
        "ghost_rules": {
            "boundary_faces": [],
            "default_type": "wall",
            "face_types": {}
        }
    }

    result = initialize_masks(grid, config)
    assert result[0].fluid_mask is False
    assert result[0].ghost_type == "wall"
    assert result[0].was_enforced is False

def test_build_simulation_grid_constructs_expected_cell_count():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "ghost_rules": {
            "boundary_faces": ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"],
            "default_type": "wall",
            "face_types": {
                "xmin": "inlet",
                "xmax": "outlet",
                "ymin": "wall",
                "ymax": "wall",
                "zmin": "wall",
                "zmax": "wall"
            }
        }
    }

    grid = build_simulation_grid(config)
    assert len(grid) == 27  # (2+1)^3

    fluid_cells = [c for c in grid if c.fluid_mask]
    ghost_cells = [c for c in grid if not c.fluid_mask]

    assert len(fluid_cells) + len(ghost_cells) == 27
    assert any(c.ghost_face == "xmin" and c.ghost_type == "inlet" for c in ghost_cells)
    assert any(c.ghost_face == "xmax" and c.ghost_type == "outlet" for c in ghost_cells)
    assert any(c.ghost_face == "ymin" and c.ghost_type == "wall" for c in ghost_cells)
    assert any(c.ghost_face == "ymax" and c.ghost_type == "wall" for c in ghost_cells)
    assert any(c.ghost_face == "zmin" and c.ghost_type == "wall" for c in ghost_cells)
    assert any(c.ghost_face == "zmax" and c.ghost_type == "wall" for c in ghost_cells)
    assert all(c.ghost_type in {"inlet", "outlet", "wall"} for c in ghost_cells)

def test_build_simulation_grid_applies_boundary_enforcement_metadata():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "ghost_rules": {
            "boundary_faces": ["xmin", "ymin", "zmin"],
            "face_types": {
                "xmin": "inlet",
                "ymin": "wall",
                "zmin": "outlet",
                "xmax": "wall",
                "ymax": "wall",
                "zmax": "wall"
            },
            "default_type": "wall"
        },
        "step_index": 99
    }

    grid = build_simulation_grid(config)
    ghost_cells = [c for c in grid if not c.fluid_mask]

    # ✅ Enforced faces
    enforced_faces = {"xmin": "inlet", "ymin": "wall", "zmin": "outlet"}
    for face, expected_type in enforced_faces.items():
        assert any(c.ghost_face == face and c.ghost_type == expected_type for c in ghost_cells)
        assert all(c.was_enforced is True for c in ghost_cells if c.ghost_face == face)
        assert all(c.ghost_source_step == 99 for c in ghost_cells if c.ghost_face == face)

    # ✅ Fallback faces
    fallback_faces = {"xmax": "wall", "ymax": "wall", "zmax": "wall"}
    for face, expected_type in fallback_faces.items():
        assert any(c.ghost_face == face and c.ghost_type == expected_type for c in ghost_cells)
        assert all(c.was_enforced is False for c in ghost_cells if c.ghost_face == face)
        assert all(c.ghost_source_step == 99 for c in ghost_cells if c.ghost_face == face)  # still tagged with step


