# tests/physics/test_viscosity.py
# 🧪 Unit tests for src/physics/viscosity.py

from src.grid_modules.cell import Cell
from src.physics.viscosity import apply_viscous_terms

def make_cell(x, y, z, velocity, fluid=True):
    return Cell(x=x, y=y, z=z, velocity=velocity, pressure=None, fluid_mask=fluid)

def test_skips_non_fluid_and_malformed_cells():
    solid = make_cell(0.0, 0.0, 0.0, velocity=[1.0, 0.0, 0.0], fluid=False)
    malformed = Cell(x=1.0, y=0.0, z=0.0, velocity=None, pressure=None, fluid_mask=True)
    config = {"fluid_properties": {"viscosity": 1.0},
              "domain_definition": {"nx": 1, "ny": 1, "nz": 1,
                                    "min_x": 0.0, "max_x": 1.0,
                                    "min_y": 0.0, "max_y": 1.0,
                                    "min_z": 0.0, "max_z": 1.0}}
    result = apply_viscous_terms([solid, malformed], dt=0.1, config=config)
    assert result[0].velocity == [1.0, 0.0, 0.0]
    assert result[1].velocity is None

def test_returns_identical_velocity_if_no_neighbors():
    center = make_cell(0.5, 0.5, 0.5, velocity=[2.0, 2.0, 2.0])
    config = {"fluid_properties": {"viscosity": 1.0},
              "domain_definition": {"nx": 2, "ny": 2, "nz": 2,
                                    "min_x": 0.0, "max_x": 1.0,
                                    "min_y": 0.0, "max_y": 1.0,
                                    "min_z": 0.0, "max_z": 1.0}}
    result = apply_viscous_terms([center], dt=0.1, config=config)
    assert result[0].velocity == [2.0, 2.0, 2.0]

def test_velocity_smoothed_toward_neighbors():
    c1 = make_cell(0.5, 0.5, 0.5, velocity=[2.0, 2.0, 2.0])
    neighbor = make_cell(0.5, 0.0, 0.5, velocity=[4.0, 4.0, 4.0])
    config = {"fluid_properties": {"viscosity": 0.5},
              "domain_definition": {"nx": 2, "ny": 2, "nz": 2,
                                    "min_x": 0.0, "max_x": 1.0,
                                    "min_y": 0.0, "max_y": 1.0,
                                    "min_z": 0.0, "max_z": 1.0}}
    result = apply_viscous_terms([c1, neighbor], dt=0.2, config=config)
    vel = result[0].velocity
    assert all(v > 2.0 for v in vel)
    assert all(v < 4.0 for v in vel)

def test_velocity_unchanged_when_viscosity_zero():
    c1 = make_cell(0.5, 0.5, 0.5, velocity=[3.0, 3.0, 3.0])
    neighbor = make_cell(0.5, 0.0, 0.5, velocity=[6.0, 6.0, 6.0])
    config = {"fluid_properties": {"viscosity": 0.0},
              "domain_definition": {"nx": 2, "ny": 2, "nz": 2,
                                    "min_x": 0.0, "max_x": 1.0,
                                    "min_y": 0.0, "max_y": 1.0,
                                    "min_z": 0.0, "max_z": 1.0}}
    result = apply_viscous_terms([c1, neighbor], dt=0.2, config=config)
    assert result[0].velocity == [3.0, 3.0, 3.0]



