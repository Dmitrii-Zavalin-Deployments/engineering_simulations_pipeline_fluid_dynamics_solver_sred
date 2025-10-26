import pytest
from unittest.mock import patch
from src.reflex.reflex_controller import apply_reflex

class MockCell:
    def __init__(self, x, y, z, velocity=None, fluid_mask=True,
                 influenced_by_ghost=False, damping_triggered=False,
                 overflow_triggered=False, cfl_exceeded=False):
        self.x = x
        self.y = y
        self.z = z
        self.velocity = velocity
        self.fluid_mask = fluid_mask
        self.influenced_by_ghost = influenced_by_ghost
        self.damping_triggered = damping_triggered
        self.overflow_triggered = overflow_triggered
        self.cfl_exceeded = cfl_exceeded

@pytest.fixture
def mock_config():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 2.0,
            "min_y": 0.0, "max_y": 2.0,
            "min_z": 0.0, "max_z": 2.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "simulation_parameters": {
            "time_step": 0.1,
            "total_time": 1.0,
            "output_interval": 1
        },
        "reflex_verbosity": "low"
    }

@patch("src.reflex.reflex_controller.detect_overflow", return_value=True)
def test_explicit_grid_injection_is_honored(mock_overflow, mock_config):
    injected_grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[5.0, 0.0, 0.0], overflow_triggered=True),
        MockCell(1.0, 0.0, 0.0, velocity=[5.0, 0.0, 0.0], damping_triggered=True),
        MockCell(0.0, 1.0, 0.0, velocity=[5.0, 0.0, 0.0], cfl_exceeded=True)
    ]

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"],
        "mutated_cells": [(0.0, 0.0, 0.0)]
    }

    result = apply_reflex(
        grid=injected_grid,
        config=mock_config,
        input_data=input_data,
        step=0,
        ghost_influence_count=0,
        pressure_solver_invoked=False,
        pressure_mutated=False,
        post_projection_divergence=1e-9
    )

    assert result["overflow_triggered_count"] == 1
    assert result["damping_triggered_count"] == 1
    assert result["cfl_exceeded_count"] == 1
    assert result["mutation_count"] == 1
    assert result["fluid_cells_modified_by_ghost"] == 0

@patch("src.reflex.reflex_controller.build_simulation_grid")
@patch("src.reflex.reflex_controller.detect_overflow", return_value=True)
def test_build_simulation_grid_patch_is_honored(mock_overflow, mock_build_grid, mock_config):
    patched_grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[9.0, 0.0, 0.0], overflow_triggered=True),
        MockCell(1.0, 0.0, 0.0, velocity=[9.0, 0.0, 0.0], damping_triggered=True)
    ]
    mock_build_grid.return_value = patched_grid

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"],
        "mutated_cells": [(0.0, 0.0, 0.0)]
    }

    result = apply_reflex(
        config=mock_config,
        input_data=input_data,
        step=1,
        ghost_influence_count=0,
        pressure_solver_invoked=True,
        pressure_mutated=True,
        post_projection_divergence=1e-8
    )

    assert result["overflow_triggered_count"] == 1
    assert result["damping_triggered_count"] == 1
    assert result["mutation_count"] == 1
    assert result["fluid_cells_modified_by_ghost"] == 0

def test_adjust_time_step_triggers_on_high_cfl_and_mutation(mock_config):
    grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[10.1, 0.0, 0.0]),  # mutated
        MockCell(1.0, 0.0, 0.0, velocity=[10.1, 0.0, 0.0]),  # mutated
        MockCell(0.0, 1.0, 0.0, velocity=[10.1, 0.0, 0.0]),  # mutated
        MockCell(1.0, 1.0, 0.0, velocity=[10.1, 0.0, 0.0])
    ]

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"],
        "mutated_cells": [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0)
        ]
    }

    result = apply_reflex(
        grid=grid,
        config=mock_config,
        input_data=input_data,
        step=2,
        ghost_influence_count=0,
        pressure_solver_invoked=True,
        pressure_mutated=True,
        post_projection_divergence=1e-8
    )

    assert result["global_cfl"] > 1.0
    assert result["mutation_density"] > 0.5
    assert result["adjusted_time_step"] < 0.1  # ✅ timestep reduced

@patch("src.reflex.reflex_controller.detect_overflow", return_value=True)
def test_overflow_triggered_count_is_flag_based(mock_overflow, mock_config):
    # ⚠️ Ensure only one cell is flagged to match assertion
    grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[1e20, 0.0, 0.0], overflow_triggered=True),
        MockCell(1.0, 0.0, 0.0, velocity=[1e20, 0.0, 0.0])  # not flagged
    ]

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"],
        "mutated_cells": [(0.0, 0.0, 0.0)]
    }

    result = apply_reflex(
        grid=grid,
        config=mock_config,
        input_data=input_data,
        step=3,
        ghost_influence_count=0,
        pressure_solver_invoked=False,
        pressure_mutated=False,
        post_projection_divergence=1e-9
    )

    assert result["overflow_detected"] is True
    assert result["overflow_triggered_count"] == 1  # ✅ only one cell flagged



