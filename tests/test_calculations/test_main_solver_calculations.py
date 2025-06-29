import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.main_solver import Simulation

@pytest.fixture
def mock_input_data():
    return {
        "simulation_parameters": {
            "total_time": 0.1,
            "time_step": 0.05,
            "output_frequency_steps": 1,
            "solver_type": "explicit"
        },
        "fluid_properties": {
            "density": 1.0,
            "viscosity": 0.01
        },
        "initial_conditions": {
            "velocity": [1.0, 0.0, 0.0],
            "pressure": 101325.0
        },
        "mesh_info": {
            "grid_shape": [2, 2, 2],
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
            "min_x": 0.0,
            "max_x": 2.0,
            "min_y": 0.0,
            "max_y": 2.0,
            "min_z": 0.0,
            "max_z": 2.0,
            "boundary_conditions": {
                "inlet": {
                    "cell_indices": [0, 1],
                    "velocity": [1.0, 0.0, 0.0],
                    "type": "dirichlet"
                }
            }
        }
    }

@patch("src.main_solver.load_input_data")
@patch("src.main_solver.save_field_snapshot")
def test_simulation_initialization_and_run(mock_save_snapshot, mock_load, mock_input_data, tmp_path):
    mock_load.return_value = mock_input_data

    sim = Simulation("input.json", str(tmp_path))

    shape = (4, 4, 4)
    sim.u = np.ones(shape)
    sim.v = np.zeros(shape)
    sim.w = np.zeros(shape)
    sim.p = np.full(shape, 101325.0)
    sim.velocity_field = np.stack((sim.u, sim.v, sim.w), axis=-1)
    sim.output_frequency_steps = 1

    sim.time_stepper.step = MagicMock(
        side_effect=lambda vel, p: (vel + 1.0, p - 1.0)
    )

    sim.run()

    assert sim.current_time == pytest.approx(0.1)
    assert sim.step_count == 2
    assert np.allclose(sim.velocity_field[..., 0], 3.0)
    assert np.allclose(sim.p, 101323.0)
    assert mock_save_snapshot.call_count >= 2

@patch("src.main_solver.initialize_grid")
@patch("src.main_solver.load_input_data")
def test_missing_boundary_indices_raises_error(mock_load, mock_init_grid, mock_input_data):
    broken = dict(mock_input_data)
    broken["mesh_info"] = {
        "grid_shape": [2, 2, 2],
        "dx": 1.0, "dy": 1.0, "dz": 1.0,
        "min_x": 0.0, "max_x": 2.0,
        "min_y": 0.0, "max_y": 2.0,
        "min_z": 0.0, "max_z": 2.0,
        "boundary_conditions": {
            "inlet": {
                "velocity": [1.0, 0.0, 0.0],
                "type": "dirichlet"
                # Intentionally missing 'cell_indices'
            }
        }
    }

    mock_load.return_value = broken
    mock_init_grid.side_effect = lambda self, _: setattr(self, "mesh_info", broken["mesh_info"])

    with pytest.raises(ValueError, match="missing 'cell_indices'"):
        Simulation("broken.json", "dummy_output")



