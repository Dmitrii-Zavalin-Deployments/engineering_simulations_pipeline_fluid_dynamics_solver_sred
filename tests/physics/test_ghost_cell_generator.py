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

def test_ghost_generated_for_x_max_tag():
    cell = make_cell(1.0, 0.5, 0.5)
    config = make_config(boundary_tags={"x_max": "outlet"})
    padded, registry = generate_ghost_cells([cell], config)

    ghosts = [c for c in padded if id(c) in registry]
    assert len(ghosts) >= 1
    ghost = ghosts[0]
    assert ghost.x > cell.x

def test_ghost_generated_for_y_and_z_faces():
    cell = make_cell(0.5, 0.0, 0.0)
    config = make_config(boundary_tags={"y_min": "wall", "z_min": "symmetry"})
    padded, registry = generate_ghost_cells([cell], config)

    ghosts = [c for c in padded if id(c) in registry]
    assert any(g.y < cell.y for g in ghosts)
    assert any(g.z < cell.z for g in ghosts)

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
    ghost_coords = [(g.x, g.y, g.z) for g in padded if id(g) in registry]
    assert any(gx < cell.x for gx, _, _ in ghost_coords)
    assert any(gy < cell.y for _, gy, _ in ghost_coords)
    assert any(gz < cell.z for _, _, gz in ghost_coords)

def test_ghost_spacing_uses_domain_resolution():
    cell = make_cell(0.0, 0.0, 0.0)
    config = make_config(xmin=0.0, xmax=2.0, nx=4, boundary_tags={"x_min": "inlet"})
    padded, registry = generate_ghost_cells([cell], config)

    dx = (2.0 - 0.0) / 4
    ghost = next(c for c in padded if id(c) in registry)
    assert abs(ghost.x - (cell.x - dx)) < 1e-6

def test_registry_includes_only_added_ghosts():
    cell = make_cell(1.0, 1.0, 1.0)
    config = make_config(boundary_tags={"x_max": "outlet"})
    padded, registry = generate_ghost_cells([cell], config)

    real_ids = {id(c) for c in padded}
    ghost_ids = set(registry)
    fluid_ids = {id(c) for c in padded if c.fluid_mask}
    assert ghost_ids.isdisjoint(fluid_ids)
    assert all(id(c) in ghost_ids for c in padded if not c.fluid_mask and c.velocity is None)



