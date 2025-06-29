import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
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
            "pressure": 100000.0
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

@patch("src.main_solver.save_field_snapshot")
@patch("src.main_solver.ExplicitSolver")
@patch("src.main_solver.initialize_fields")
@patch("src.main_solver.initialize_grid")
@patch("src.main_solver.initialize_simulation_parameters")
@patch("src.main_solver.load_input_data")
def test_simulation_initialization_and_run(
    mock_load,
    mock_init_params,
    mock_init_grid,
    mock_init_fields,
    mock_solver_class,
    mock_save_snapshot,
    mock_input_data,
    tmp_path
):
    mock_load.return_value = mock_input_data
    sim = Simulation("dummy_input.json", str(tmp_path))

    # Inject velocity & pressure arrays after fields are created
    sim.u = np.ones((4, 4, 4))
    sim.v = np.zeros((4, 4, 4))
    sim.w = np.zeros((4, 4, 4))
    sim.p = np.full((4, 4, 4), 100000.0)
    sim.velocity_field = np.stack((sim.u, sim.v, sim.w), axis=-1)

    # Patch step method to evolve pressure field
    sim.time_stepper.step = MagicMock(
        side_effect=lambda u, p: (u + 1.0, p - 1.0)
    )

    sim.run()

    # Ensure snapshot saved at both step 0 and 2
    assert mock_save_snapshot.call_count >= 2
    assert sim.step_count == 2
    assert sim.current_time == pytest.approx(0.1)
    np.testing.assert_allclose(sim.p, 99998.0)
    np.testing.assert_allclose(sim.velocity_field[..., 0], 3.0)

@patch("src.main_solver.load_input_data")
def test_missing_boundary_indices_raises_error(mock_load, mock_input_data):
    faulty_data = dict(mock_input_data)
    faulty_data["mesh_info"]["boundary_conditions"]["inlet"].pop("cell_indices")
    mock_load.return_value = faulty_data

    with pytest.raises(ValueError, match="missing 'cell_indices'"):
        Simulation("invalid.json", "dummy_output")




