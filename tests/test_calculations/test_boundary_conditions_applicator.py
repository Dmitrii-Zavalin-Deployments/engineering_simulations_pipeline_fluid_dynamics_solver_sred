import numpy as np
import pytest
from src.physics.boundary_conditions_applicator import apply_boundary_conditions

@pytest.fixture
def mesh_info_dirichlet():
    return {
        "grid_shape": (5, 5, 5),
        "boundary_conditions": {
            "inlet": {
                "type": "dirichlet",
                "apply_to": ["velocity", "pressure"],
                "velocity": [1.0, 0.0, 0.0],
                "pressure": 5.0,
                "cell_indices": [[0, 2, 2]],
                "boundary_dim": 0
            }
        }
    }

@pytest.fixture
def mesh_info_neumann():
    return {
        "grid_shape": (5, 5, 5),
        "boundary_conditions": {
            "outlet": {
                "type": "neumann",
                "apply_to": ["velocity"],
                "cell_indices": [[4, 2, 2]],
                "interior_neighbor_offset": [-1, 0, 0],
                "boundary_dim": 0
            }
        }
    }

def test_apply_dirichlet_velocity_and_pressure(mesh_info_dirichlet):
    velocity = np.zeros((5, 5, 5, 3), dtype=np.float64)
    pressure = np.zeros((5, 5, 5), dtype=np.float64)
    fluid_properties = {}

    v_out, p_out = apply_boundary_conditions(velocity.copy(), pressure.copy(), fluid_properties, mesh_info_dirichlet, is_tentative_step=False)

    assert np.allclose(v_out[0, 2, 2], [1.0, 0.0, 0.0])
    assert p_out[0, 2, 2] == pytest.approx(5.0)

def test_skip_pressure_if_tentative_step(mesh_info_dirichlet):
    velocity = np.zeros((5, 5, 5, 3), dtype=np.float64)
    pressure = np.ones((5, 5, 5), dtype=np.float64) * 3.14
    fluid_properties = {}

    _, p_out = apply_boundary_conditions(velocity.copy(), pressure.copy(), fluid_properties, mesh_info_dirichlet, is_tentative_step=True)

    # Pressure should remain unchanged
    assert p_out[0, 2, 2] == pytest.approx(3.14)

def test_apply_neumann_velocity_from_interior(mesh_info_neumann):
    velocity = np.zeros((5, 5, 5, 3), dtype=np.float64)
    velocity[3, 2, 2] = [2.0, 0.0, 0.0]  # Interior neighbor

    pressure = np.zeros((5, 5, 5), dtype=np.float64)
    fluid_properties = {}

    v_out, _ = apply_boundary_conditions(velocity.copy(), pressure, fluid_properties, mesh_info_neumann, is_tentative_step=False)

    # Boundary should copy value from neighbor
    assert np.allclose(v_out[4, 2, 2], [2.0, 0.0, 0.0])

def test_missing_boundary_conditions_returns_untouched():
    velocity = np.random.rand(5, 5, 5, 3)
    pressure = np.random.rand(5, 5, 5)
    fluid_properties = {}
    mesh_info = {"grid_shape": (5, 5, 5)}  # missing 'boundary_conditions'

    v_out, p_out = apply_boundary_conditions(velocity.copy(), pressure.copy(), fluid_properties, mesh_info, is_tentative_step=False)
    assert np.allclose(v_out, velocity)
    assert np.allclose(p_out, pressure)

def test_invalid_field_type_triggers_fallback(mesh_info_dirichlet):
    pressure = np.zeros((5, 5, 5), dtype=np.float64)
    fluid_properties = {}

    # velocity has wrong dtype
    velocity = np.zeros((5, 5, 5, 3), dtype=np.int32)
    v_out, p_out = apply_boundary_conditions(velocity, pressure, fluid_properties, mesh_info_dirichlet, is_tentative_step=False)

    assert isinstance(v_out, np.ndarray)
    assert isinstance(p_out, np.ndarray)
    assert v_out.dtype == velocity.dtype  # unchanged due to early return



