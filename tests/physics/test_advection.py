# tests/test_advection.py
# 🧪 Unit tests for compute_advection — verifies ghost exclusion, symmetry skipping, and velocity evolution

import pytest
from src.grid_modules.cell import Cell
from src.physics.advection import compute_advection

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=1.0, fluid_mask=fluid)

@pytest.fixture
def domain_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 3, "ny": 1, "nz": 1
        }
    }

def test_advection_applies_euler_to_fluid_cells(domain_config):
    cell1 = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    cell2 = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    grid = [cell1, cell2]
    updated = compute_advection(grid, dt=0.1, config=domain_config)
    assert isinstance(updated, list)
    assert len(updated) == 2
    assert updated[1].velocity != [0.0, 0.0, 0.0]

def test_advection_skips_ghost_cells(domain_config):
    cell1 = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    cell2 = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    grid = [cell1, cell2]
    ghost_registry = {id(cell2)}
    updated = compute_advection(grid, dt=0.1, config=domain_config, ghost_registry=ghost_registry)
    assert updated[0].velocity == [1.0, 0.0, 0.0]

def test_advection_skips_symmetry_ghost_cells(domain_config):
    cell = make_cell(1.0, 0.0, 0.0, [0.0, 1.0, 0.0])
    grid = [cell]
    ghost_registry = {id(cell)}
    ghost_metadata = {id(cell): {"boundary_type": "symmetry"}}
    updated = compute_advection(grid, dt=0.1, config=domain_config, ghost_registry=ghost_registry, ghost_metadata=ghost_metadata)
    assert updated == []

def test_advection_handles_nonfluid_cells(domain_config):
    fluid = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0], fluid=True)
    solid = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0], fluid=False)
    grid = [solid, fluid]
    updated = compute_advection(grid, dt=0.1, config=domain_config)
    assert len(updated) == 2
    assert updated[0].velocity == [1.0, 0.0, 0.0]  # unchanged
    assert updated[1].velocity == [0.0, 0.0, 0.0]  # preserved if no upstream

def test_advection_tracks_velocity_mutation_count(capsys, domain_config):
    cell1 = make_cell(0.0, 0.0, 0.0, [1.0, 0.0, 0.0])
    cell2 = make_cell(1.0, 0.0, 0.0, [0.5, 0.0, 0.0])
    grid = [cell1, cell2]
    compute_advection(grid, dt=0.1, config=domain_config)
    output = capsys.readouterr().out
    assert "velocity mutations" in output
    assert "Advection applied to 2 physical cells" in output

def test_advection_with_malformed_velocity_skips_mutation(domain_config):
    fluid = Cell(x=1.0, y=0.0, z=0.0, velocity="bad", pressure=1.0, fluid_mask=True)
    result = compute_advection([fluid], dt=0.1, config=domain_config)
    assert result[0].velocity is None

def test_advection_returns_empty_if_all_ghosts(domain_config):
    ghost = make_cell(1.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    grid = [ghost]
    ghost_registry = {id(ghost)}
    result = compute_advection(grid, dt=0.1, config=domain_config, ghost_registry=ghost_registry)
    assert result == []

def test_advection_returns_unchanged_velocity_if_no_neighbors(domain_config):
    cell = make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    result = compute_advection([cell], dt=0.1, config=domain_config)
    assert result[0].velocity == [0.0, 0.0, 0.0]



