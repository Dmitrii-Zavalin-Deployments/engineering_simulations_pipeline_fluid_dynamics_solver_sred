# tests/physics/test_ghost_cell_generator.py
# 🧪 Validates ghost creation across boundary faces, fluid proximity, and registry accuracy

import pytest
from src.grid_modules.cell import Cell
from src.physics.ghost_cell_generator import generate_ghost_cells

def make_fluid(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 1.0, 1.0], pressure=10.0, fluid_mask=True)

def make_solid(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=[0.0, 0.0, 0.0], pressure=None, fluid_mask=False)

@pytest.fixture
def default_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 2.0, "ny": 2,
            "min_z": 0.0, "max_z": 2.0, "nz": 2
        },
        "boundary_conditions": {
            "apply_faces": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
            "velocity": [5.0, 0.0, 0.0],
            "pressure": 99.0,
            "no_slip": False
        }
    }

def test_ghosts_created_for_all_faces(default_config):
    fluid = make_fluid(0.0, 0.0, 0.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    faces = set(meta["face"] for meta in registry.values())
    assert faces == {"x_min", "y_min", "z_min"}
    assert len(registry) == 3

def test_no_slip_enforces_zero_velocity(default_config):
    default_config["boundary_conditions"]["no_slip"] = True
    fluid = make_fluid(0.0, 0.0, 0.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    for ghost in padded:
        if not ghost.fluid_mask:
            assert ghost.velocity == [0.0, 0.0, 0.0]

def test_velocity_and_pressure_enforced(default_config):
    fluid = make_fluid(2.0, 2.0, 2.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    for ghost in padded:
        if not ghost.fluid_mask:
            assert ghost.velocity == [5.0, 0.0, 0.0]
            assert ghost.pressure == 99.0

def test_registry_contains_correct_metadata(default_config):
    fluid = make_fluid(0.0, 0.0, 0.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    for meta in registry.values():
        assert isinstance(meta["origin"], tuple)
        assert "face" in meta
        assert isinstance(meta["coordinate"], tuple)
        assert isinstance(meta["velocity"], list)

def test_no_ghosts_for_non_boundary_fluid(default_config):
    fluid = make_fluid(1.0, 1.0, 1.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    assert len(registry) == 0
    assert padded == [fluid]

def test_multiple_fluid_cells_create_multiple_ghosts(default_config):
    cells = [
        make_fluid(0.0, 0.0, 0.0),   # x_min, y_min, z_min
        make_fluid(2.0, 2.0, 2.0),   # x_max, y_max, z_max
        make_fluid(1.0, 1.0, 1.0)    # center → no ghosts
    ]
    padded, registry = generate_ghost_cells(cells, default_config)
    assert len(registry) == 6
    face_counts = {face: 0 for face in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]}
    for meta in registry.values():
        face_counts[meta["face"]] += 1
    assert all(count == 1 for count in face_counts.values())

def test_solid_cells_do_not_trigger_ghosts(default_config):
    solid = make_solid(0.0, 0.0, 0.0)
    padded, registry = generate_ghost_cells([solid], default_config)
    assert padded == [solid]
    assert registry == {}

def test_missing_boundary_config_defaults_safely():
    fluid = make_fluid(0.0, 0.0, 0.0)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 2.0, "ny": 2,
            "min_z": 0.0, "max_z": 2.0, "nz": 2
        }
    }
    padded, registry = generate_ghost_cells([fluid], config)
    assert isinstance(padded, list)
    assert isinstance(registry, dict)

def test_invalid_enforced_pressure_does_not_crash(default_config):
    default_config["boundary_conditions"]["pressure"] = "invalid"
    fluid = make_fluid(0.0, 0.0, 0.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    for ghost in padded:
        if not ghost.fluid_mask:
            assert ghost.pressure is None

def test_invalid_velocity_fallbacks(default_config):
    default_config["boundary_conditions"]["velocity"] = "not-a-vector"
    fluid = make_fluid(0.0, 0.0, 0.0)
    padded, registry = generate_ghost_cells([fluid], default_config)
    for ghost in padded:
        if not ghost.fluid_mask:
            assert isinstance(ghost.velocity, list)