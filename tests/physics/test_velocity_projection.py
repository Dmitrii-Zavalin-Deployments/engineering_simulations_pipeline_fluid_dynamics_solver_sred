import pytest
from src.physics.velocity_projection import apply_pressure_velocity_projection
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=None, fluid_mask=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask)

def test_velocity_projection_applies_gradient_subtraction():
    # Setup: central cell with neighbors having pressure gradient
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    dx = dy = dz = 1.0
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[2.0, 2.0, 2.0], pressure=5.0)
    p_plus_x = make_cell(2.0, 1.0, 1.0, pressure=7.0)
    p_minus_x = make_cell(0.0, 1.0, 1.0, pressure=3.0)
    p_plus_y = make_cell(1.0, 2.0, 1.0, pressure=6.0)
    p_minus_y = make_cell(1.0, 0.0, 1.0, pressure=4.0)
    p_plus_z = make_cell(1.0, 1.0, 2.0, pressure=8.0)
    p_minus_z = make_cell(1.0, 1.0, 0.0, pressure=2.0)

    grid = [fluid, p_plus_x, p_minus_x, p_plus_y, p_minus_y, p_plus_z, p_minus_z]
    projected = apply_pressure_velocity_projection(grid, config)

    updated = [c for c in projected if c.fluid_mask][0]
    expected_grad = [(7.0 - 3.0) / (2.0 * dx), (6.0 - 4.0) / (2.0 * dy), (8.0 - 2.0) / (2.0 * dz)]
    expected_velocity = [2.0 - g for g in expected_grad]

    assert updated.velocity == pytest.approx(expected_velocity, abs=1e-6)

def test_skips_nonfluid_cells():
    solid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0], pressure=5.0, fluid_mask=False)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }
    projected = apply_pressure_velocity_projection([solid], config)
    assert projected[0].velocity == [1.0, 1.0, 1.0]
    assert projected[0].pressure == 5.0

def test_skips_cells_missing_pressure_or_velocity():
    fluid_missing_pressure = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0], pressure=None)
    fluid_missing_velocity = make_cell(1.0, 1.0, 1.0, velocity=None, pressure=5.0)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }

    projected1 = apply_pressure_velocity_projection([fluid_missing_pressure], config)
    projected2 = apply_pressure_velocity_projection([fluid_missing_velocity], config)

    assert getattr(projected1[0], "projection_skipped", False) is True
    assert getattr(projected2[0], "projection_skipped", False) is True

def test_projection_preserves_pressure_and_coordinates():
    fluid = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0], pressure=5.0)
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        }
    }
    projected = apply_pressure_velocity_projection([fluid], config)
    cell = projected[0]
    assert cell.x == 1.0 and cell.y == 1.0 and cell.z == 1.0
    assert cell.pressure == 5.0
    assert isinstance(cell.velocity, list)



