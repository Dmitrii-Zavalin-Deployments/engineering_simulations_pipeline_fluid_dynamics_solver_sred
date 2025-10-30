# tests/test_ghost_cell_generator.py

import pytest
from src.physics.ghost_cell_generator import generate_ghost_cells, EPSILON
from src.grid_modules.cell import Cell

# 🔧 Utility: Create a fluid cell
def make_fluid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=True)

# 🔧 Utility: Create a solid cell
def make_solid_cell(x, y, z):
    return Cell(x=x, y=y, z=z, velocity=None, pressure=None, fluid_mask=False)

# 🔧 Fixtures
@pytest.fixture
def domain():
    return {
        "nx": 2, "ny": 2, "nz": 2,
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0
    }

@pytest.fixture
def ghost_rules():
    return {
        "boundary_faces": ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"],
        "face_types": {
            "x_min": "wall", "x_max": "inlet",
            "y_min": "wall", "y_max": "outlet",
            "z_min": "wall", "z_max": "wall"
        },
        "default_type": "wall"
    }

@pytest.fixture
def boundary_conditions():
    return [
        {
            "apply_faces": ["x_min", "x_max"],
            "apply_to": ["velocity", "pressure"],
            "type": "dirichlet",
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 101325
        },
        {
            "apply_faces": ["y_min", "y_max"],
            "apply_to": ["pressure"],
            "type": "dirichlet",
            "pressure": 101000
        },
        {
            "apply_faces": ["z_min", "z_max"],
            "apply_to": ["velocity"],
            "type": "dirichlet",
            "velocity": [0.0, 0.0, 0.0]
        }
    ]

# ✅ Test: Ghost creation at all six faces
def test_ghost_creation_all_faces(domain, ghost_rules, boundary_conditions):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    dy = (domain["max_y"] - domain["min_y"]) / domain["ny"]
    dz = (domain["max_z"] - domain["min_z"]) / domain["nz"]

    grid = [
        make_fluid_cell(domain["min_x"] + 0.5 * dx, 0.5, 0.5),  # x_min
        make_fluid_cell(domain["max_x"] - 0.5 * dx, 0.5, 0.5),  # x_max
        make_fluid_cell(0.5, domain["min_y"] + 0.5 * dy, 0.5),  # y_min
        make_fluid_cell(0.5, domain["max_y"] - 0.5 * dy, 0.5),  # y_max
        make_fluid_cell(0.5, 0.5, domain["min_z"] + 0.5 * dz),  # z_min
        make_fluid_cell(0.5, 0.5, domain["max_z"] - 0.5 * dz)   # z_max
    ]

    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": boundary_conditions
    }

    padded, registry = generate_ghost_cells(grid, config, debug=False)

    assert len(padded) == len(grid) + 6
    assert len(registry) == 6
    assert set(entry["face"] for entry in registry.values()) == set(ghost_rules["boundary_faces"])

# ✅ Test: Solid cells are excluded
def test_excludes_solid_cells(domain, ghost_rules, boundary_conditions):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    fluid = make_fluid_cell(domain["min_x"] + 0.5 * dx, 0.5, 0.5)
    solid = make_solid_cell(0.5, 0.5, 0.5)
    grid = [fluid, solid]

    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": boundary_conditions
    }

    padded, registry = generate_ghost_cells(grid, config, debug=False)

    assert len(padded) == 3  # 2 input cells + 1 ghost
    assert len(registry) == 1

# ✅ Test: Missing top-level config keys
@pytest.mark.parametrize("missing_key", ["domain_definition", "ghost_rules", "boundary_conditions"])
def test_missing_top_level_keys(missing_key, domain, ghost_rules, boundary_conditions):
    grid = [make_fluid_cell(0.5, 0.5, 0.5)]
    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": boundary_conditions
    }
    del config[missing_key]

    with pytest.raises(KeyError):
        generate_ghost_cells(grid, config)

# ✅ Test: Missing domain keys
def test_missing_domain_keys(domain, ghost_rules, boundary_conditions):
    del domain["nx"]
    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": boundary_conditions
    }
    grid = [make_fluid_cell(0.5, 0.5, 0.5)]

    with pytest.raises(KeyError):
        generate_ghost_cells(grid, config)

# ✅ Test: Missing ghost_rules keys
@pytest.mark.parametrize("missing_key", ["boundary_faces", "face_types", "default_type"])
def test_missing_ghost_rule_keys(missing_key, domain, ghost_rules, boundary_conditions):
    del ghost_rules[missing_key]
    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": boundary_conditions
    }
    grid = [make_fluid_cell(0.5, 0.5, 0.5)]

    with pytest.raises(KeyError):
        generate_ghost_cells(grid, config)

# ✅ Test: Missing velocity or pressure in Dirichlet BC
def test_missing_dirichlet_values(domain, ghost_rules):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    grid = [make_fluid_cell(domain["min_x"] + 0.5 * dx, 0.5, 0.5)]
    bad_conditions = [
        {
            "apply_faces": ["x_min"],
            "apply_to": ["velocity"],
            "type": "dirichlet"
            # missing velocity
        }
    ]
    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": bad_conditions
    }

    with pytest.raises(ValueError):
        generate_ghost_cells(grid, config)

# ✅ Test: Registry metadata integrity
def test_registry_metadata(domain, ghost_rules, boundary_conditions):
    dx = (domain["max_x"] - domain["min_x"]) / domain["nx"]
    grid = [make_fluid_cell(domain["min_x"] + 0.5 * dx, 0.5, 0.5)]
    config = {
        "domain_definition": domain,
        "ghost_rules": ghost_rules,
        "boundary_conditions": boundary_conditions
    }

    _, registry = generate_ghost_cells(grid, config, debug=False)
    ghost = list(registry.values())[0]

    assert ghost["face"] == "x_min"
    assert ghost["velocity"] == [1.0, 0.0, 0.0]
    assert ghost["pressure"] == 101325
    assert ghost["enforcement"]["velocity"] is True
    assert ghost["enforcement"]["pressure"] is True
