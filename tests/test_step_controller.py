import pytest
from unittest import mock
from src.step_controller import evolve_step
from src.grid_modules.cell import Cell

@pytest.fixture
def mock_cell():
    return mock.Mock(spec=Cell, fluid_mask=True)

@pytest.fixture
def input_data():
    return {
        "domain_definition": {
            "min_x": 0.0, "max_x": 1.0,
            "min_y": 0.0, "max_y": 1.0,
            "min_z": 0.0, "max_z": 1.0,
            "nx": 2, "ny": 2, "nz": 2
        },
        "fluid_properties": {"density": 1.0, "viscosity": 0.001},
        "initial_conditions": {"initial_velocity": [0.0, 0.0, 0.0], "initial_pressure": 101325.0},
        "simulation_parameters": {"time_step": 0.01, "total_time": 0.1, "output_interval": 1},
        "geometry_definition": {
            "geometry_mask_flat": [1] * 8,
            "geometry_mask_shape": [2, 2, 2],
            "mask_encoding": {"fluid": 1, "solid": 0},
            "flattening_order": "x-major"
        },
        "default_timestep": 0.01
    }

@pytest.fixture
def config():
    return {
        "reflex_verbosity": "high",
        "ghost_adjacency_depth": 2
    }

@pytest.fixture
def sim_config(input_data):
    return input_data

def test_evolve_step_runs_all_modules(monkeypatch, input_data, config, sim_config, mock_cell):
    grid = [mock_cell for _ in range(8)]

    monkeypatch.setattr("src.step_controller.generate_ghost_cells", lambda g, i: (g, {"ghost_0": mock.Mock(), "ghost_1": mock.Mock()}))
    monkeypatch.setattr("src.step_controller.log_ghost_summary", lambda ghosts: None)
    monkeypatch.setattr("src.step_controller.apply_boundary_conditions", lambda g, ghosts, i: g)
    monkeypatch.setattr("src.step_controller.apply_ghost_influence", lambda g, s, verbose, radius: 5)
    monkeypatch.setattr("src.step_controller.compute_divergence_stats", lambda g, s, label, step_index, output_folder, config=None: {"max": 0.01})
    monkeypatch.setattr("src.step_controller.suggest_timestep", lambda delta_path, trace_path, base_dt, reflex_score=None: 0.005)
    monkeypatch.setattr("src.step_controller.solve_navier_stokes_step", lambda g, i, s: (g, {"pressure_mutated": True, "projection_passes": 2}))
    monkeypatch.setattr("src.step_controller.apply_reflex", lambda g, i, s, ghost_influence_count, config, sim_config, pressure_solver_invoked, pressure_mutated, post_projection_divergence: {"reflex_score": 4.2})
    monkeypatch.setattr("src.step_controller.inject_diagnostics", lambda m, ghosts, grid, spacing: m)

    evolved_grid, metadata = evolve_step(grid, input_data, step=0, config=config, sim_config=sim_config)

    assert isinstance(evolved_grid, list)
    assert all(isinstance(c, Cell) or isinstance(c, mock.Mock) for c in evolved_grid)
    assert metadata["reflex_score"] == 4.2
    assert metadata["ghost_influence_count"] == 5
    assert metadata["pressure_mutated"] is True
    assert metadata["projection_passes"] == 2
    assert metadata["adaptive_timestep"] == 0.005
    assert metadata["boundary_condition_applied"] is True
    assert "ghost_registry" in metadata

def test_evolve_step_excludes_only_solid_cells(monkeypatch, input_data, config, sim_config):
    fluid = mock.Mock(spec=Cell, fluid_mask=True)
    solid = mock.Mock(spec=Cell, fluid_mask=False)
    grid = [fluid, solid, fluid]

    monkeypatch.setattr("src.step_controller.generate_ghost_cells", lambda g, i: (g, {"ghost_0": mock.Mock()}))
    monkeypatch.setattr("src.step_controller.apply_boundary_conditions", lambda g, ghosts, i: g)
    monkeypatch.setattr("src.step_controller.apply_ghost_influence", lambda g, s, verbose, radius: sum(1 for c in g if getattr(c, "fluid_mask", False)))
    monkeypatch.setattr("src.step_controller.compute_divergence_stats", lambda *a, **kw: {"max": 0.0})
    monkeypatch.setattr("src.step_controller.suggest_timestep", lambda *a, **kw: 0.01)
    monkeypatch.setattr("src.step_controller.solve_navier_stokes_step", lambda g, i, s: (g, {"pressure_mutated": True, "projection_passes": 1}))
    monkeypatch.setattr("src.step_controller.apply_reflex", lambda *a, **kw: {"reflex_score": 1.0})
    monkeypatch.setattr("src.step_controller.inject_diagnostics", lambda m, ghosts, grid, spacing: m)

    evolved_grid, metadata = evolve_step(grid, input_data, step=1, config=config, sim_config=sim_config)
    assert metadata["ghost_influence_count"] == 2

def test_evolve_step_handles_missing_config(monkeypatch, input_data, sim_config, mock_cell):
    grid = [mock_cell for _ in range(4)]

    monkeypatch.setattr("src.step_controller.generate_ghost_cells", lambda g, i: (g, {"ghost_0": mock.Mock()}))
    monkeypatch.setattr("src.step_controller.apply_boundary_conditions", lambda g, ghosts, i: g)
    monkeypatch.setattr("src.step_controller.apply_ghost_influence", lambda g, s, verbose, radius: 1)
    monkeypatch.setattr("src.step_controller.compute_divergence_stats", lambda *a, **kw: {"max": 0.0})
    monkeypatch.setattr("src.step_controller.suggest_timestep", lambda *a, **kw: 0.01)
    monkeypatch.setattr("src.step_controller.solve_navier_stokes_step", lambda g, i, s: (g, {"pressure_mutated": True, "projection_passes": 1}))
    monkeypatch.setattr("src.step_controller.apply_reflex", lambda *a, **kw: {"reflex_score": 1.0})
    monkeypatch.setattr("src.step_controller.inject_diagnostics", lambda m, ghosts, grid, spacing: m)

    with pytest.raises(ValueError):
        evolve_step(grid, input_data, step=2, config=None, sim_config=sim_config)
