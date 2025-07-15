# tests/physics/test_pressure_projection.py
# 🧪 Validates pressure Poisson solve and ghost-aware projection across fluid cells

import pytest
from src.grid_modules.cell import Cell
from src.physics.pressure_projection import extract_ghost_coords, solve_pressure_poisson

def make_cell(x, y, z, velocity, pressure, fluid=True, ghost_face=None):
    cell = Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)
    if ghost_face:
        setattr(cell, "ghost_face", ghost_face)
    return cell

@pytest.fixture
def config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "pressure_solver": {
            "method": "jacobi",
            "max_iterations": 20,
            "tolerance": 1e-5
        }
    }

def test_extract_ghost_coords_identifies_tagged_cells():
    c1 = make_cell(0.0, 0.0, 0.0, None, None, fluid=False, ghost_face="x_min")
    c2 = make_cell(1.0, 0.0, 0.0, None, None, fluid=False)
    c3 = make_cell(2.0, 0.0, 0.0, None, None, fluid=True)
    result = extract_ghost_coords([c1, c2, c3])
    assert result == {(0.0, 0.0, 0.0)}

def test_pressure_projection_applies_mutated_pressure(config):
    fluid = Cell(x=1.0, y=0.0, z=0.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
    ghost = make_cell(0.0, 0.0, 0.0, None, 99.0, fluid=False, ghost_face="x_min")
    grid = [ghost, fluid]
    divergence = [1.0]
    projected, mutated = solve_pressure_poisson(grid, divergence, config)
    assert isinstance(projected, list)
    assert mutated is True
    assert isinstance(projected[1].pressure, float)
    assert projected[1].pressure != 0.0
    assert isinstance(projected[1].velocity, list)

def test_pressure_projection_handles_equal_pressure(config):
    fluid = Cell(x=1.0, y=0.0, z=0.0, velocity=[0.0, 0.0, 0.0], pressure=10.0, fluid_mask=True)
    ghost = make_cell(0.0, 0.0, 0.0, None, 10.0, fluid=False, ghost_face="x_min")
    grid = [ghost, fluid]
    divergence = [0.0]
    projected, mutated = solve_pressure_poisson(grid, divergence, config)
    assert isinstance(projected, list)
    assert mutated is False

def test_pressure_projection_raises_if_mismatch(config):
    fluid1 = Cell(x=1.0, y=0.0, z=0.0, velocity=[0.0]*3, pressure=1.0, fluid_mask=True)
    fluid2 = Cell(x=2.0, y=0.0, z=0.0, velocity=[0.0]*3, pressure=1.0, fluid_mask=True)
    grid = [fluid1, fluid2]
    divergence = [0.5]
    with pytest.raises(ValueError, match="Divergence list length"):
        solve_pressure_poisson(grid, divergence, config)

def test_unsupported_method_raises(config):
    config["pressure_solver"]["method"] = "not_supported"
    fluid = Cell(x=1.0, y=0.0, z=0.0, velocity=[0.0]*3, pressure=0.0, fluid_mask=True)
    with pytest.raises(ValueError, match="Unknown or unsupported pressure solver"):
        solve_pressure_poisson([fluid], [0.0], config)

def test_projection_preserves_nonfluid_cells(config):
    fluid = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], 0.0, fluid=True)
    ghost = make_cell(0.0, 0.0, 0.0, None, None, fluid=False, ghost_face="x_min")
    projected, _ = solve_pressure_poisson([fluid, ghost], [0.0], config)
    assert not projected[0].fluid_mask
    assert projected[0].velocity is None
    assert projected[0].pressure is None

def test_projection_rebuilds_velocity(config):
    fluid = make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0], 0.0, fluid=True)
    ghost = make_cell(0.0, 0.0, 0.0, None, 99.0, fluid=False, ghost_face="x_min")
    grid = [ghost, fluid]
    projected, _ = solve_pressure_poisson(grid, [0.5], config)
    assert isinstance(projected[1].velocity, list)
    assert len(projected[1].velocity) == 3