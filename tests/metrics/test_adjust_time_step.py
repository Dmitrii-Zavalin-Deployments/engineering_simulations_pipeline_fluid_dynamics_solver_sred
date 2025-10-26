import pytest
from src.reflex.reflex_logic import adjust_time_step

class MockCell:
    def __init__(self, x, y, z, velocity=None, fluid_mask=True, cfl_exceeded=False):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity or [0.0, 0.0, 0.0]
        self.fluid_mask = fluid_mask
        self.cfl_exceeded = cfl_exceeded

@pytest.fixture
def base_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 1, "ny": 1, "nz": 1
        },
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1
        }
    }

def test_timestep_reduction_on_high_cfl(base_config):
    grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[100.0, 0.0, 0.0], cfl_exceeded=True),
        MockCell(1.0, 0.0, 0.0, velocity=[100.0, 0.0, 0.0], cfl_exceeded=True)
    ]

    input_data = {
        "domain_definition": base_config["domain_definition"],
        "simulation_parameters": base_config["simulation_parameters"],
        "mutated_cells": []
    }

    reduced_dt = adjust_time_step(grid, input_data)
    assert reduced_dt < 0.1
    assert reduced_dt > 0.0

def test_timestep_reduction_on_high_mutation_density(base_config):
    # ✅ Patch: Increase velocity to ensure CFL > 1.0
    grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[10.5, 0.0, 0.0]),
        MockCell(1.0, 0.0, 0.0, velocity=[10.5, 0.0, 0.0]),
        MockCell(0.0, 1.0, 0.0, velocity=[10.5, 0.0, 0.0]),
        MockCell(1.0, 1.0, 0.0, velocity=[1.0, 0.0, 0.0])
    ]

    input_data = {
        "domain_definition": base_config["domain_definition"],
        "simulation_parameters": base_config["simulation_parameters"],
        "mutated_cells": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
        ]
    }

    reduced_dt = adjust_time_step(grid, input_data)
    assert reduced_dt < 0.1
    assert reduced_dt > 0.0

def test_no_timestep_reduction_on_low_cfl_and_mutation(base_config):
    grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[0.1, 0.0, 0.0]),
        MockCell(1.0, 0.0, 0.0, velocity=[0.1, 0.0, 0.0])
    ]

    input_data = {
        "domain_definition": base_config["domain_definition"],
        "simulation_parameters": base_config["simulation_parameters"],
        "mutated_cells": [(0.0, 0.0, 0.0)]
    }

    unchanged_dt = adjust_time_step(grid, input_data)
    assert unchanged_dt == 0.1



