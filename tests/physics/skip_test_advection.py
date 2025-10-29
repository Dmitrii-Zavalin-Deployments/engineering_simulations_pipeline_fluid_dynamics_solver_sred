# tests/physics/test_advection.py
# ✅ Validation suite for src/physics/advection.py

import pytest
from src.physics.advection import compute_advection
from src.grid_modules.cell import Cell

def make_cell(x, y, z, velocity=None, pressure=0.0, fluid_mask=True):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity or [0.0, 0.0, 0.0],
        pressure=pressure,
        fluid_mask=fluid_mask
    )

def test_advection_skips_solid_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        }
    }
    solid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid_mask=False)
    result = compute_advection([solid], dt=0.1, config=config)
    assert result[0].velocity == [1.0, 0.0, 0.0]
    assert not hasattr(result[0], "transport_triggered")

def test_advection_applies_euler_update_on_fluid_cell():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "step_index": 5
    }

    # Central cell with neighbors in ±x
    center = make_cell(1.0, 1.0, 1.0, velocity=[1.0, 0.0, 0.0])
    x_plus = make_cell(2.0, 1.0, 1.0, velocity=[2.0, 0.0, 0.0])
    x_minus = make_cell(0.0, 1.0, 1.0, velocity=[0.0, 0.0, 0.0])

    grid = [center, x_plus, x_minus]
    result = compute_advection(grid, dt=0.1, config=config)

    updated = result[0]
    assert isinstance(updated.velocity, list)
    assert updated.velocity[0] < 1.0  # velocity reduced by transport
    assert updated.mutation_source == "advection"
    assert updated.mutation_step == 5
    assert updated.transport_triggered is True

def test_advection_handles_missing_neighbors_gracefully():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        }
    }

    cell = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 2.0, 3.0])
    result = compute_advection([cell], dt=0.1, config=config)
    updated = result[0]
    assert updated.velocity == [1.0, 2.0, 3.0]  # ✅ velocity unchanged due to zero-gradient fallback
    assert updated.transport_triggered is True  # ✅ reflex tag confirms processing

def test_advection_preserves_nonfluid_cells_unchanged():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        }
    }

    fluid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 1.0, 1.0], fluid_mask=True)
    solid = make_cell(1.0, 0.0, 0.0, velocity=[9.0, 9.0, 9.0], fluid_mask=False)

    result = compute_advection([fluid, solid], dt=0.1, config=config)
    assert result[1].velocity == [9.0, 9.0, 9.0]
    assert not hasattr(result[1], "transport_triggered")

def test_advection_handles_multiple_fluid_cells():
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "step_index": 3
    }

    grid = [
        make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, velocity=[0.0, 1.0, 0.0]),
        make_cell(0.0, 1.0, 0.0, velocity=[0.0, 0.0, 1.0]),
        make_cell(1.0, 1.0, 0.0, velocity=[1.0, 1.0, 1.0])
    ]

    result = compute_advection(grid, dt=0.05, config=config)
    assert len(result) == 4
    for cell in result:
        assert cell.transport_triggered is True
        assert cell.mutation_source == "advection"
        assert cell.mutation_step == 3



