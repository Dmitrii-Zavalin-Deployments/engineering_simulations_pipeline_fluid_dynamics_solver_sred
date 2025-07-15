# tests/physics/test_viscosity.py
# 🧪 Validates Laplacian-based viscosity smoothing and fluid-only mutation logic

import pytest
from src.grid_modules.cell import Cell
from src.physics.viscosity import apply_viscous_terms

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(
        x=x, y=y, z=z,
        velocity=velocity,
        pressure=1.0,
        fluid_mask=fluid
    )

@pytest.fixture
def config_3x1x1():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 3.0, "nx": 3,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "fluid_properties": {
            "viscosity": 0.5
        }
    }

def test_viscosity_applies_laplacian(config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 2.0, 2.0])
    ]
    result = apply_viscous_terms(grid, dt=0.1, config=config_3x1x1)
    mutated_velocity = result[1].velocity
    assert mutated_velocity == [1.0, 1.0, 1.0]
    avg = [(0.0 + 2.0)/2, (0.0 + 2.0)/2, (0.0 + 2.0)/2]
    expected = [1.0 + 0.5 * 0.1 * (avg[i] - 1.0) for i in range(3)]
    assert mutated_velocity == pytest.approx(expected)

def test_viscosity_skips_nonfluid_cells(config_3x1x1):
    solid = make_cell(1.0, 0.0, 0.0, [9.9, 9.9, 9.9], fluid=False)
    result = apply_viscous_terms([solid], dt=0.1, config=config_3x1x1)
    assert result[0].velocity == [9.9, 9.9, 9.9]

def test_viscosity_skips_malformed_velocity(config_3x1x1):
    broken = Cell(x=1.0, y=0.0, z=0.0, velocity="bad", pressure=1.0, fluid_mask=True)
    result = apply_viscous_terms([broken], dt=0.1, config=config_3x1x1)
    assert result[0].velocity is None

def test_viscosity_preserves_if_no_neighbors(config_3x1x1):
    solo = make_cell(1.0, 0.0, 0.0, [5.0, 5.0, 5.0])
    result = apply_viscous_terms([solo], dt=0.1, config=config_3x1x1)
    assert result[0].velocity == [5.0, 5.0, 5.0]

def test_zero_viscosity_produces_no_mutation(config_3x1x1):
    config_3x1x1["fluid_properties"]["viscosity"] = 0.0
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [1.0, 1.0, 1.0]),
        make_cell(2.0, 0.0, 0.0, [2.0, 2.0, 2.0])
    ]
    result = apply_viscous_terms(grid, dt=0.1, config=config_3x1x1)
    assert result[1].velocity == [1.0, 1.0, 1.0]

def test_viscosity_mutation_count_prints(capsys, config_3x1x1):
    grid = [
        make_cell(0.0, 0.0, 0.0, [0.0, 0.0, 0.0]),
        make_cell(1.0, 0.0, 0.0, [9.0, 9.0, 9.0]),
        make_cell(2.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    ]
    apply_viscous_terms(grid, dt=0.1, config=config_3x1x1)
    output = capsys.readouterr().out
    assert "velocity mutations" in output

def test_viscosity_spatial_spacing(config_3x1x1):
    config = {
        "domain_definition": {
            "min_x": 0.0, "max_x": 4.0, "nx": 4,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        },
        "fluid_properties": {
            "viscosity": 1.0
        }
    }
    cell = make_cell(1.0, 0.0, 0.0, [4.0, 4.0, 4.0])
    neighbor = make_cell(2.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    grid = [cell, neighbor]
    result = apply_viscous_terms(grid, dt=0.2, config=config)
    assert result[0].velocity != [4.0, 4.0, 4.0]



