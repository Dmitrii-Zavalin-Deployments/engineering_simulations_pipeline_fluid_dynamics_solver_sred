# tests/physics/test_viscosity.py
# ✅ Unit tests for Laplacian-based velocity damping in apply_viscous_terms

import pytest
from src.grid_modules.cell import Cell
from src.physics.viscosity import apply_viscous_terms

def make_fluid_cell(x, y, z, velocity):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=velocity,
        pressure=1.0,
        fluid_mask=True
    )

def make_solid_cell(x, y, z):
    return Cell(
        x=x,
        y=y,
        z=z,
        velocity=None,
        pressure=None,
        fluid_mask=False
    )

def config_with_viscosity(viscosity=0.1):
    return {
        "fluid_properties": {"viscosity": viscosity},
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 3.0, "ny": 3,
            "min_z": 0.0, "max_z": 3.0, "nz": 3
        }
    }

def test_viscosity_applied_to_fluid_cell_with_neighbors():
    c0 = make_fluid_cell(1.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    c1 = make_fluid_cell(2.0, 1.0, 1.0, [2.0, 0.0, 0.0])
    c2 = make_fluid_cell(0.0, 1.0, 1.0, [0.0, 0.0, 0.0])
    grid = [c0, c1, c2]
    updated = apply_viscous_terms(grid, dt=0.5, config=config_with_viscosity(0.2))

    v0 = updated[0].velocity
    # Laplacian = ((2.0 + 0.0)/2 - 1.0) = 0.0 → no change if symmetrical
    assert isinstance(v0, list)
    assert abs(v0[0] - 1.0) < 0.5  # Should slightly shift toward neighbor avg

def test_viscosity_skips_cells_without_neighbors():
    c0 = make_fluid_cell(1.0, 1.0, 1.0, [3.0, 3.0, 3.0])
    grid = [c0]
    updated = apply_viscous_terms(grid, dt=1.0, config=config_with_viscosity(0.5))
    assert updated[0].velocity == [3.0, 3.0, 3.0]

def test_viscosity_preserves_solid_cells():
    c0 = make_solid_cell(0.0, 0.0, 0.0)
    updated = apply_viscous_terms([c0], dt=1.0, config=config_with_viscosity(0.1))
    assert updated[0].velocity is None
    assert updated[0].fluid_mask is False

def test_viscosity_applied_with_zero_viscosity():
    c0 = make_fluid_cell(1.0, 1.0, 1.0, [2.0, 0.0, 0.0])
    c1 = make_fluid_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    grid = [c0, c1]
    config = config_with_viscosity(viscosity=0.0)
    updated = apply_viscous_terms(grid, dt=0.5, config=config)
    assert updated[0].velocity == [2.0, 0.0, 0.0]
    assert updated[1].velocity == [1.0, 0.0, 0.0]

def test_velocity_damping_direction():
    c0 = make_fluid_cell(1.0, 1.0, 1.0, [5.0, 0.0, 0.0])
    c1 = make_fluid_cell(2.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    c2 = make_fluid_cell(0.0, 1.0, 1.0, [1.0, 0.0, 0.0])
    grid = [c0, c1, c2]
    updated = apply_viscous_terms(grid, dt=1.0, config=config_with_viscosity(0.25))

    damped = updated[0].velocity[0]
    assert damped < 5.0
    assert damped > 1.0  # Should move toward average of neighbors

def test_viscosity_vector_component_integrity():
    c0 = make_fluid_cell(1.0, 1.0, 1.0, [2.0, 4.0, 6.0])
    c1 = make_fluid_cell(2.0, 1.0, 1.0, [0.0, 0.0, 0.0])
    c2 = make_fluid_cell(0.0, 1.0, 1.0, [0.0, 0.0, 0.0])
    grid = [c0, c1, c2]
    updated = apply_viscous_terms(grid, dt=0.1, config=config_with_viscosity(0.2))

    v_new = updated[0].velocity
    assert len(v_new) == 3
    assert all(isinstance(v, float) for v in v_new)



