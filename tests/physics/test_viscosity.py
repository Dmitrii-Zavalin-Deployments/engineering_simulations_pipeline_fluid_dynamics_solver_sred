import pytest
from src.physics.viscosity import apply_viscous_terms
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def test_viscosity_applies_laplacian_update():
    # Central fluid cell surrounded by 6 neighbors with higher velocity
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0])
    neighbors = [
        make_cell(2.0, 1.0, 1.0, velocity=[2.0, 2.0, 2.0]),
        make_cell(0.0, 1.0, 1.0, velocity=[2.0, 2.0, 2.0]),
        make_cell(1.0, 2.0, 1.0, velocity=[2.0, 2.0, 2.0]),
        make_cell(1.0, 0.0, 1.0, velocity=[2.0, 2.0, 2.0]),
        make_cell(1.0, 1.0, 2.0, velocity=[2.0, 2.0, 2.0]),
        make_cell(1.0, 1.0, 0.0, velocity=[2.0, 2.0, 2.0]),
    ]
    grid = [fluid] + neighbors

    config = {
        "fluid_properties": {"viscosity": 0.5},
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 3.0,
            "min_z": 0.0, "max_z": 3.0,
            "nx": 3, "ny": 3, "nz": 3
        },
        "step_index": 10
    }

    updated = apply_viscous_terms(grid, dt=0.1, config=config)
    central = [c for c in updated if c.x == 1.0 and c.y == 1.0 and c.z == 1.0][0]

    assert central.velocity != [1.0, 1.0, 1.0]
    assert central.mutation_source == "viscosity"
    assert central.mutation_step == 10
    assert central.damping_triggered is True

def test_viscosity_skips_solid_cells():
    solid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0], fluid_mask=False)
    config = {
        "fluid_properties": {"viscosity": 1.0},
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }
    updated = apply_viscous_terms([solid], dt=0.1, config=config)
    assert updated[0].velocity == [1.0, 1.0, 1.0]
    assert not hasattr(updated[0], "damping_triggered")

def test_viscosity_skips_cells_with_no_neighbors():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0])
    config = {
        "fluid_properties": {"viscosity": 1.0},
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }
    updated = apply_viscous_terms([fluid], dt=0.1, config=config)
    assert updated[0].velocity == [1.0, 1.0, 1.0]
    assert not hasattr(updated[0], "damping_triggered")

def test_viscosity_handles_zero_viscosity():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0])
    neighbor = make_cell(2.0, 1.0, 1.0, velocity=[2.0, 2.0, 2.0])
    grid = [fluid, neighbor]

    config = {
        "fluid_properties": {"viscosity": 0.0},
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 3.0,
            "min_z": 0.0, "max_z": 3.0,
            "nx": 3, "ny": 3, "nz": 3
        }
    }

    updated = apply_viscous_terms(grid, dt=0.1, config=config)
    assert updated[0].velocity == [1.0, 1.0, 1.0]
    assert updated[0].damping_triggered is False

def test_viscosity_preserves_pressure_and_coordinates():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0], pressure=5.0)
    neighbor = make_cell(2.0, 1.0, 1.0, velocity=[2.0, 2.0, 2.0])
    grid = [fluid, neighbor]

    config = {
        "fluid_properties": {"viscosity": 1.0},
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 3.0,
            "min_z": 0.0, "max_z": 3.0,
            "nx": 3, "ny": 3, "nz": 3
        }
    }

    updated = apply_viscous_terms(grid, dt=0.1, config=config)
    cell = updated[0]
    assert cell.x == 1.0 and cell.y == 1.0 and cell.z == 1.0
    assert cell.pressure == 5.0



