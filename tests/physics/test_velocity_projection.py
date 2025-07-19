# tests/physics/test_velocity_projection.py
# 🧪 Unit tests for src/physics/velocity_projection.py

from src.grid_modules.cell import Cell
from src.physics.velocity_projection import apply_pressure_velocity_projection

def make_cell(x, y, z, velocity, pressure, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=pressure, fluid_mask=fluid)

def test_projection_applies_gradient_subtraction():
    c1 = make_cell(0.0, 0.0, 0.0, velocity=[5.0, 5.0, 5.0], pressure=1.0)
    c2 = make_cell(1.0, 0.0, 0.0, velocity=[5.0, 5.0, 5.0], pressure=3.0)
    c_mid = make_cell(0.5, 0.0, 0.0, velocity=[5.0, 5.0, 5.0], pressure=2.0)
    config = {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1,
                              "min_x": 0.0, "max_x": 1.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0}
    }
    projected = apply_pressure_velocity_projection([c1, c2, c_mid], config)
    mid = [c for c in projected if c.x == 0.5][0]
    expected_grad_x = (3.0 - 1.0) / (2.0 * 0.5)
    assert abs(mid.velocity[0] - (5.0 - expected_grad_x)) < 1e-6

def test_non_fluid_cells_are_skipped():
    fluid = make_cell(0.5, 0.0, 0.0, velocity=[1.0, 1.0, 1.0], pressure=2.0)
    ghost = make_cell(0.0, 0.0, 0.0, velocity=None, pressure=None, fluid=False)
    config = {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1,
                              "min_x": 0.0, "max_x": 1.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0}
    }
    projected = apply_pressure_velocity_projection([fluid, ghost], config)
    assert projected[1].fluid_mask is False
    assert projected[1].velocity is None

def test_skips_cell_if_pressure_or_velocity_missing():
    fluid_missing_pressure = make_cell(1.0, 0.0, 0.0, velocity=[2.0, 2.0, 2.0], pressure=None)
    fluid_missing_velocity = make_cell(0.0, 0.0, 0.0, velocity=None, pressure=1.0)
    config = {
        "domain_definition": {"nx": 2, "ny": 1, "nz": 1,
                              "min_x": 0.0, "max_x": 1.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0}
    }
    projected = apply_pressure_velocity_projection([fluid_missing_pressure, fluid_missing_velocity], config)
    for c in projected:
        assert c.velocity in ([2.0, 2.0, 2.0], None)
        assert c.pressure in (1.0, None)

def test_gradient_skips_dimension_if_neighbor_missing():
    c_mid = make_cell(0.5, 0.5, 0.5, velocity=[3.0, 3.0, 3.0], pressure=1.0)
    config = {
        "domain_definition": {"nx": 2, "ny": 2, "nz": 2,
                              "min_x": 0.0, "max_x": 1.0,
                              "min_y": 0.0, "max_y": 1.0,
                              "min_z": 0.0, "max_z": 1.0}
    }
    projected = apply_pressure_velocity_projection([c_mid], config)
    updated = projected[0]
    assert updated.velocity == [3.0, 3.0, 3.0]  # No gradient subtraction without neighbors



