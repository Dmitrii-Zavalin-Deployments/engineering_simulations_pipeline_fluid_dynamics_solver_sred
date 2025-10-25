import pytest
from src.physics.pressure_projection import solve_pressure_poisson
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def test_pressure_projection_applies_to_fluid_cells_only():
    fluid1 = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], pressure=0.0)
    fluid2 = make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0], pressure=0.0)
    solid = make_cell(2.0, 0.0, 0.0, fluid_mask=False)
    grid = [fluid1, fluid2, solid]
    divergence = [0.1, -0.2]

    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        },
        "pressure_solver": {"method": "jacobi"},
        "step_index": 5
    }

    projected, mutated, registry = solve_pressure_poisson(grid, divergence, config, verbose=False)
    fluid_cells = [c for c in projected if c.fluid_mask]
    solid_cells = [c for c in projected if not c.fluid_mask]

    assert len(fluid_cells) == 2
    assert all(isinstance(c.pressure, float) for c in fluid_cells)
    assert all(c.velocity is not None for c in fluid_cells)
    assert all(c.fluid_mask for c in fluid_cells)
    assert all(c.pressure_mutated or c.pressure_delta == 0.0 for c in fluid_cells)
    assert all(c.mutation_source == "pressure_solver" for c in fluid_cells if c.pressure_mutated)
    assert all(c.mutation_step == 5 for c in fluid_cells if c.pressure_mutated)

    assert len(solid_cells) == 1
    assert solid_cells[0].pressure is None
    assert solid_cells[0].velocity is None

def test_raises_error_if_divergence_length_mismatch():
    fluid = make_cell(0.0, 0.0, 0.0)
    grid = [fluid]
    divergence = [0.1, 0.2]  # too long

    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "pressure_solver": {"method": "jacobi"}
    }

    with pytest.raises(ValueError, match="Divergence list length"):
        solve_pressure_poisson(grid, divergence, config)

def test_raises_error_for_unknown_solver_method():
    fluid = make_cell(0.0, 0.0, 0.0)
    grid = [fluid]
    divergence = [0.0]

    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "pressure_solver": {"method": "unsupported"}
    }

    with pytest.raises(ValueError, match="Unknown or unsupported pressure solver method"):
        solve_pressure_poisson(grid, divergence, config)

def test_mutation_tagging_and_delta_tracking():
    fluid = make_cell(0.0, 0.0, 0.0, pressure=0.0)
    grid = [fluid]
    divergence = [0.5]

    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "pressure_solver": {"method": "jacobi"},
        "step_index": 42
    }

    projected, mutated, registry = solve_pressure_poisson(grid, divergence, config)
    cell = projected[0]
    assert mutated is True
    assert cell.pressure_mutated is True
    assert cell.mutation_source == "pressure_solver"
    assert cell.mutation_step == 42
    assert cell.pressure_delta > 0.0

def test_ghost_influence_tagging():
    ghost = make_cell(0.0, 0.0, 0.0, fluid_mask=False)
    fluid = make_cell(0.5, 0.0, 0.0, pressure=0.0)
    grid = [ghost, fluid]
    divergence = [0.1]

    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 1, "nz": 1
        },
        "pressure_solver": {"method": "jacobi"},
        "step_index": 7
    }

    projected, mutated, registry = solve_pressure_poisson(grid, divergence, config)
    fluid_cell = [c for c in projected if c.fluid_mask][0]
    assert fluid_cell.influenced_by_ghost is True
    assert fluid_cell.mutation_triggered_by == "ghost_influence"



