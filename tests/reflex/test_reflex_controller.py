import pytest
from unittest.mock import patch
from src.reflex.reflex_controller import apply_reflex

class MockCell:
    def __init__(self, x, y, z, velocity=None, fluid_mask=True, influenced_by_ghost=False,
                 damping_triggered=False, overflow_triggered=False, cfl_exceeded=False):
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

@patch("src.reflex.reflex_controller.build_simulation_grid")
def test_reflex_basic_metrics(mock_build_grid, mock_config):
    mock_grid = [
        MockCell(0.0, 0.0, 0.0, velocity=[0.1, 0.0, 0.0]),
        MockCell(1.0, 0.0, 0.0, velocity=[0.1, 0.0, 0.0], influenced_by_ghost=True),
        MockCell(0.0, 1.0, 0.0, velocity=[0.1, 0.0, 0.0], damping_triggered=True),
        MockCell(1.0, 1.0, 0.0, velocity=[1e20, 0.0, 0.0], overflow_triggered=True),
        MockCell(0.0, 0.0, 1.0, velocity=[0.1, 0.0, 0.0], cfl_exceeded=True),
        MockCell(1.0, 0.0, 1.0, fluid_mask=False)
    ]
    mock_build_grid.return_value = mock_grid

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"],
        "mutated_cells": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    }

    result = apply_reflex(
        config=mock_config,
        input_data=input_data,
        step=1,
        ghost_influence_count=1,
        pressure_solver_invoked=True,
        pressure_mutated=True,
        post_projection_divergence=1e-9
    )

    assert result["max_velocity"] >= 0.0
    assert result["max_divergence"] >= 0.0
    assert result["global_cfl"] >= 0.0
    assert result["overflow_detected"] is True
    assert result["damping_enabled"] is True
    assert result["adjusted_time_step"] > 0.0
    assert result["projection_passes"] >= 0
    assert result["divergence_zero"] is True
    assert result["projection_skipped"] in [True, False]
    assert result["fluid_cells_modified_by_ghost"] == 1
    assert result["mutation_count"] == 2
    assert result["mutation_density"] > 0.0
    assert result["damping_triggered_count"] == 1
    assert result["overflow_triggered_count"] == 1
    assert result["cfl_exceeded_count"] == 1
    assert isinstance(result["reflex_score"], float)

@patch("src.reflex.reflex_controller.build_simulation_grid")
def test_reflex_handles_empty_mutation_list(mock_build_grid, mock_config):
    mock_grid = [MockCell(0.0, 0.0, 0.0), MockCell(1.0, 0.0, 0.0)]
    mock_build_grid.return_value = mock_grid

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"]
    }

    result = apply_reflex(
        config=mock_config,
        input_data=input_data,
        step=2,
        ghost_influence_count=0,
        pressure_solver_invoked=False,
        pressure_mutated=False,
        post_projection_divergence=1e-5
    )

    assert result["mutation_count"] == 0
    assert result["mutation_density"] == 0.0
    assert result["pressure_mutated"] is False
    assert result["pressure_solver_invoked"] is False

@patch("src.reflex.reflex_controller.build_simulation_grid")
def test_reflex_tags_suppression_when_no_influence(mock_build_grid, mock_config):
    mock_grid = [
        MockCell(0.0, 0.0, 0.0),
        MockCell(1.0, 0.0, 0.0),
        MockCell(0.0, 1.0, 0.0)
    ]
    mock_build_grid.return_value = mock_grid

    input_data = {
        "domain_definition": mock_config["domain_definition"],
        "simulation_parameters": mock_config["simulation_parameters"],
        "mutated_cells": [(0.0, 0.0, 0.0)]
    }

    result = apply_reflex(
        config=mock_config,
        input_data=input_data,
        step=3,
        ghost_influence_count=2,
        pressure_solver_invoked=True,
        pressure_mutated=True,
        post_projection_divergence=1e-6
    )

    assert isinstance(result["suppression_zones"], list)
    assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in result["suppression_zones"])



