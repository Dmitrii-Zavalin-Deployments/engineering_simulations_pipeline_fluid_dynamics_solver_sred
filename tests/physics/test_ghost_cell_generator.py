import pytest
from collections import Counter
from src.physics.ghost_cell_generator import generate_ghost_cells
from src.grid_modules.cell import Cell

def minimal_domain():
    return {
        "min_x": 0.0, "max_x": 1.0,
        "min_y": 0.0, "max_y": 1.0,
        "min_z": 0.0, "max_z": 1.0,
        "nx": 1, "ny": 1, "nz": 1
    }

def base_config(boundary_conditions, ghost_rules):
    return {
        "domain_definition": minimal_domain(),
        "boundary_conditions": boundary_conditions,
        "ghost_rules": ghost_rules
    }

def fluid_cell(x=0.0, y=0.0, z=0.0):
    return Cell(x=x, y=y, z=z, velocity=[1.0, 0.0, 0.0], pressure=133.0, fluid_mask=True)

# 🧪 Test: Dirichlet inlet
def test_inlet_dirichlet_velocity_and_pressure():
    config = base_config(
        boundary_conditions=[{
            "role": "inlet",
            "type": "dirichlet",
            "apply_to": ["velocity", "pressure"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 133.0,
            "apply_faces": ["x_min"]
        }],
        ghost_rules={
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "inlet"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(x=0.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.velocity == [1.0, 0.0, 0.0]
    assert ghost.pressure == 133.0

# 🧪 Test: Neumann outlet
def test_outlet_neumann_pressure_only():
    config = base_config(
        boundary_conditions=[{
            "role": "outlet",
            "type": "neumann",
            "apply_to": ["pressure"],
            "apply_faces": ["x_max"]
        }],
        ghost_rules={
            "boundary_faces": ["x_max"],
            "face_types": {"x_max": "outlet"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(x=1.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.velocity is None
    assert ghost.pressure is None

# 🧪 Test: Wall with no-slip velocity
def test_wall_dirichlet_velocity_only():
    config = base_config(
        boundary_conditions=[{
            "role": "wall",
            "type": "dirichlet",
            "apply_to": ["velocity"],
            "velocity": [0.0, 0.0, 0.0],
            "apply_faces": ["y_min"]
        }],
        ghost_rules={
            "boundary_faces": ["y_min"],
            "face_types": {"y_min": "wall"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(y=0.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.velocity == [0.0, 0.0, 0.0]
    assert ghost.pressure is None

# 🧪 Test: Missing velocity for Dirichlet
def test_missing_velocity_dirichlet_raises():
    config = base_config(
        boundary_conditions=[{
            "role": "inlet",
            "type": "dirichlet",
            "apply_to": ["velocity"],
            "apply_faces": ["x_min"]
        }],
        ghost_rules={
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "inlet"},
            "default_type": "wall"
        }
    )
    with pytest.raises(ValueError, match="Missing velocity for face 'x_min'"):
        generate_ghost_cells([fluid_cell(x=0.0)], config, debug=False)

# 🧪 Test: Missing pressure for Dirichlet
def test_missing_pressure_dirichlet_raises():
    config = base_config(
        boundary_conditions=[{
            "role": "inlet",
            "type": "dirichlet",
            "apply_to": ["pressure"],
            "apply_faces": ["x_min"]
        }],
        ghost_rules={
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "inlet"},
            "default_type": "wall"
        }
    )
    with pytest.raises(ValueError, match="Missing pressure for face 'x_min'"):
        generate_ghost_cells([fluid_cell(x=0.0)], config, debug=False)

# 🧪 Test: No boundary condition defined
def test_missing_boundary_condition_raises():
    config = base_config(
        boundary_conditions=[],
        ghost_rules={
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "inlet"},
            "default_type": "wall"
        }
    )
    with pytest.raises(ValueError, match="No boundary condition defined for face 'x_min'"):
        generate_ghost_cells([fluid_cell(x=0.0)], config, debug=False)

# 🧪 Test: Multi-face ghost creation and registry count
def test_multi_face_ghost_creation_and_registry():
    config = base_config(
        boundary_conditions=[
            {
                "role": "inlet",
                "type": "dirichlet",
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 133.0,
                "apply_faces": ["x_min"]
            },
            {
                "role": "wall",
                "type": "dirichlet",
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "apply_faces": ["y_min"]
            },
            {
                "role": "outlet",
                "type": "neumann",
                "apply_to": ["pressure"],
                "apply_faces": ["x_max"]
            }
        ],
        ghost_rules={
            "boundary_faces": ["x_min", "y_min", "x_max"],
            "face_types": {
                "x_min": "inlet",
                "y_min": "wall",
                "x_max": "outlet"
            },
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(x=0.0, y=0.0), fluid_cell(x=1.0)], config, debug=False)
    ghosts = [c for c in grid if not c.fluid_mask]
    assert len(ghosts) == 4
    face_counts = Counter([registry[id(g)]["face"] for g in ghosts])
    assert face_counts["x_min"] == 1
    assert face_counts["x_max"] == 1
    assert face_counts["y_min"] == 2

# 🧪 Test: Debug-enabled ghost generation path coverage
def test_debug_logging_path_exercised(capfd):
    config = base_config(
        boundary_conditions=[{
            "role": "inlet",
            "type": "dirichlet",
            "apply_to": ["velocity", "pressure"],
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 133.0,
            "apply_faces": ["x_min"]
        }],
        ghost_rules={
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "inlet"},
            "default_type": "wall"
        }
    )
    generate_ghost_cells([fluid_cell(x=0.0)], config, debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] 📘 [ghost_gen] Ghost rule config:" in out
    assert "[DEBUG] 🧱 Ghost created" in out
    assert "[DEBUG] 📊 Ghost generation complete" in out

# 🧪 Test: Debug-enabled ghost generation path coverage
def test_debug_logging_path_exercised(capfd):
    config = base_config(
        boundary_conditions=[
            {
                "role": "inlet",
                "type": "dirichlet",
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 133.0,
                "apply_faces": ["x_min"]
            },
            {
                "role": "wall",
                "type": "dirichlet",
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "apply_faces": ["y_min"]
            }
        ],
        ghost_rules={
            "boundary_faces": ["x_min", "y_min"],
            "face_types": {
                "x_min": "inlet",
                "y_min": "wall"
            },
            "default_type": "wall"
        }
    )
    generate_ghost_cells([fluid_cell(x=0.0, y=0.0)], config, debug=True)
    out, _ = capfd.readouterr()
    assert "[DEBUG] 📘 [ghost_gen] Ghost rule config:" in out
    assert "[DEBUG] 🔍 Evaluating fluid[0]" in out
    assert "[DEBUG] 🧱 Ghost created @ (-1.00, 0.00, 0.00)" in out
    assert "[DEBUG] 🧱 Ghost created @ (0.00, -1.00, 0.00)" in out
    assert "[DEBUG] 📊 Ghost generation complete" in out
    assert "[DEBUG]    x_min: 1 ghosts" in out
    assert "[DEBUG]    y_min: 1 ghosts" in out

# 🧪 Test: Debug-enabled ghost generation path coverage (full trace)
def test_debug_logging_path_exercised_full(capfd):
    config = base_config(
        boundary_conditions=[
            {
                "role": "inlet",
                "type": "dirichlet",
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 133.0,
                "apply_faces": ["x_min"]
            },
            {
                "role": "wall",
                "type": "dirichlet",
                "apply_to": ["velocity"],
                "velocity": [0.0, 0.0, 0.0],
                "apply_faces": ["y_min"]
            }
        ],
        ghost_rules={
            "boundary_faces": ["x_min", "y_min"],
            "face_types": {
                "x_min": "inlet",
                "y_min": "wall"
            },
            "default_type": "wall"
        }
    )
    generate_ghost_cells([fluid_cell(x=0.0, y=0.0)], config, debug=True)
    out, _ = capfd.readouterr()

    # ✅ Confirm all debug lines are printed
    assert "[DEBUG] 📘 [ghost_gen] Ghost rule config:" in out
    assert "[DEBUG] 🔍 Evaluating fluid[0]" in out
    assert "[DEBUG] 🧱 Ghost created @ (-1.00, 0.00, 0.00)" in out
    assert "[DEBUG] 🧱 Ghost created @ (0.00, -1.00, 0.00)" in out
    assert "[DEBUG] 📊 Ghost generation complete → total: 2" in out
    assert "[DEBUG]    x_min: 1 ghosts" in out
    assert "[DEBUG]    y_min: 1 ghosts" in out

# 🧪 Test: Neumann velocity enforcement
def test_neumann_velocity_only():
    config = base_config(
        boundary_conditions=[{
            "role": "outlet",
            "type": "neumann",
            "apply_to": ["velocity"],
            "apply_faces": ["x_max"]
        }],
        ghost_rules={
            "boundary_faces": ["x_max"],
            "face_types": {"x_max": "outlet"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(x=1.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.velocity is None
    assert ghost.pressure is None

# 🧪 Test: Non-fluid cell is skipped
def test_non_fluid_cell_skipped():
    config = base_config(
        boundary_conditions=[{
            "role": "inlet",
            "type": "dirichlet",
            "apply_to": ["velocity"],
            "velocity": [1.0, 0.0, 0.0],
            "apply_faces": ["x_min"]
        }],
        ghost_rules={
            "boundary_faces": ["x_min"],
            "face_types": {"x_min": "inlet"},
            "default_type": "wall"
        }
    )
    non_fluid = Cell(x=0.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=False)
    grid, registry = generate_ghost_cells([non_fluid], config, debug=False)
    ghosts = [c for c in grid if not c.fluid_mask and c is not non_fluid]
    assert len(ghosts) == 0

# 🧪 Test: y_max face triggers ghost creation
def test_y_max_face_triggered():
    config = base_config(
        boundary_conditions=[{
            "role": "wall",
            "type": "dirichlet",
            "apply_to": ["velocity"],
            "velocity": [0.0, 0.0, 0.0],
            "apply_faces": ["y_max"]
        }],
        ghost_rules={
            "boundary_faces": ["y_max"],
            "face_types": {"y_max": "wall"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(y=1.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.ghost_face == "y_max"

# 🧪 Test: z_min face triggers ghost creation
def test_z_min_face_triggered():
    config = base_config(
        boundary_conditions=[{
            "role": "wall",
            "type": "dirichlet",
            "apply_to": ["velocity"],
            "velocity": [0.0, 0.0, 0.0],
            "apply_faces": ["z_min"]
        }],
        ghost_rules={
            "boundary_faces": ["z_min"],
            "face_types": {"z_min": "wall"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(z=0.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.ghost_face == "z_min"

# 🧪 Test: z_max face triggers ghost creation
def test_z_max_face_triggered():
    config = base_config(
        boundary_conditions=[{
            "role": "wall",
            "type": "dirichlet",
            "apply_to": ["velocity"],
            "velocity": [0.0, 0.0, 0.0],
            "apply_faces": ["z_max"]
        }],
        ghost_rules={
            "boundary_faces": ["z_max"],
            "face_types": {"z_max": "wall"},
            "default_type": "wall"
        }
    )
    grid, registry = generate_ghost_cells([fluid_cell(z=1.0)], config, debug=False)
    ghost = [c for c in grid if not c.fluid_mask][0]
    assert ghost.ghost_face == "z_max"



