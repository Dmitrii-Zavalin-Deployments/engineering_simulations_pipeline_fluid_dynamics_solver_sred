# tests/physics/test_ghost_cell_generator.py
# 🧱 Unit tests for ghost cell generator — domain-edge padding coverage

from src.physics.ghost_cell_generator import generate_ghost_cells
from src.grid_modules.cell import Cell

def make_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=1.0, fluid_mask=True)

def make_config(xmin=0.0, xmax=1.0, nx=1, boundary_tags={}):
    return {
        "domain_definition": {
            "min_x": xmin, "max_x": xmax, "nx": nx,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "boundary_conditions": boundary_tags
    }

def test_ghost_generated_for_x_min_tag():
    cell = make_cell(0.0, 0.5, 0.5)
    config = make_config(boundary_tags={"x_min": "inlet"})
    padded, registry = generate_ghost_cells([cell], config)

    ghosts = [c for c in padded if id(c) in registry]
    assert len(ghosts) >= 1
    ghost = ghosts[0]
    assert ghost.x < cell.x
    assert ghost.velocity is None
    assert ghost.pressure is None
    assert ghost.fluid_mask is False
    assert registry[id(ghost)]["face"] == "x_min"
    assert hasattr(ghost, "ghost_face")
    assert ghost.ghost_face == "x_min"

def test_ghost_generated_for_x_max_tag():
    cell = make_cell(1.0, 0.5, 0.5)
    config = make_config(boundary_tags={"x_max": "outlet"})
    padded, registry = generate_ghost_cells([cell], config)

    ghosts = [c for c in padded if id(c) in registry]
    assert len(ghosts) >= 1
    ghost = ghosts[0]
    assert ghost.x > cell.x
    assert registry[id(ghost)]["face"] == "x_max"
    assert ghost.ghost_face == "x_max"

def test_ghost_generated_for_y_and_z_faces():
    cell = make_cell(0.5, 0.0, 0.0)
    config = make_config(boundary_tags={"y_min": "wall", "z_min": "symmetry"})
    padded, registry = generate_ghost_cells([cell], config)

    ghosts = [c for c in padded if id(c) in registry]
    assert any(g.y < cell.y and g.ghost_face == "y_min" for g in ghosts)
    assert any(g.z < cell.z and g.ghost_face == "z_min" for g in ghosts)

def test_no_ghosts_without_boundary_tags():
    cell = make_cell(0.0, 0.0, 0.0)
    config = make_config(boundary_tags={})
    padded, registry = generate_ghost_cells([cell], config)

    assert padded == [cell]
    assert len(registry) == 0

def test_multiple_ghosts_from_multi_face_alignment():
    cell = make_cell(0.0, 0.0, 0.0)
    config = make_config(boundary_tags={
        "x_min": "inlet",
        "y_min": "wall",
        "z_min": "symmetry"
    })
    padded, registry = generate_ghost_cells([cell], config)
    assert len(registry) == 3
    ghost_faces = {meta["face"] for meta in registry.values()}
    assert ghost_faces == {"x_min", "y_min", "z_min"}

def test_ghost_spacing_uses_domain_resolution():
    cell = make_cell(0.0, 0.0, 0.0)
    config = make_config(xmin=0.0, xmax=2.0, nx=4, boundary_tags={"x_min": "inlet"})
    padded, registry = generate_ghost_cells([cell], config)

    dx = (2.0 - 0.0) / 4
    ghost = next(c for c in padded if id(c) in registry)
    assert abs(ghost.x - (cell.x - dx)) < 1e-6
    assert ghost.ghost_face == "x_min"

def test_registry_metadata_matches_cell_attributes():
    cell = make_cell(1.0, 1.0, 1.0)
    config = make_config(boundary_tags={"x_max": "outlet"})
    padded, registry = generate_ghost_cells([cell], config)

    ghosts = [c for c in padded if id(c) in registry]
    for ghost in ghosts:
        meta = registry[id(ghost)]
        assert "face" in meta
        assert "origin" in meta
        assert meta["face"] == ghost.ghost_face
        assert isinstance(meta["origin"], tuple)
        assert len(meta["origin"]) == 3

def test_registry_excludes_physical_cells():
    cell = make_cell(0.0, 0.0, 0.0)
    config = make_config(boundary_tags={"x_min": "inlet"})
    padded, registry = generate_ghost_cells([cell], config)

    physical_ids = {id(c) for c in padded if c.fluid_mask}
    ghost_ids = set(registry.keys())
    assert physical_ids.isdisjoint(ghost_ids)



