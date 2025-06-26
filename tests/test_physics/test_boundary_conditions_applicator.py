import numpy as np
import pytest
from src.physics.boundary_conditions_applicator import apply_boundary_conditions


def dummy_mesh(grid_shape=(8, 8, 8)):
    return {
        "grid_shape": grid_shape,
        "boundary_conditions": {}
    }


def dummy_fields(grid_shape=(8, 8, 8)):
    velocity = np.zeros((*grid_shape, 3), dtype=np.float64)
    pressure = np.zeros(grid_shape, dtype=np.float64)
    return velocity, pressure


def test_invalid_velocity_type():
    velocity = "not-an-array"
    pressure = np.zeros((4, 4, 4), dtype=np.float64)
    mesh = dummy_mesh((4, 4, 4))
    out_vel, out_p = apply_boundary_conditions(velocity, pressure, {}, mesh, True)
    assert isinstance(out_vel, str)


def test_missing_boundary_conditions_key():
    velocity, pressure = dummy_fields((4, 4, 4))
    mesh = {"grid_shape": (4, 4, 4)}  # No 'boundary_conditions' key
    out_vel, out_p = apply_boundary_conditions(velocity, pressure, {}, mesh, False)
    assert np.all(out_vel == 0.0)
    assert np.all(out_p == 0.0)


def test_skips_empty_cell_indices(capfd):
    v, p = dummy_fields()
    mesh = dummy_mesh()
    mesh["boundary_conditions"] = {
        "empty_face": {
            "type": "dirichlet",
            "cell_indices": np.zeros((0, 3), dtype=int),
            "velocity": np.array([1.0, 0.0, 0.0]),
            "pressure": 42.0,
            "apply_to": ["velocity", "pressure"],
            "boundary_dim": 0,
            "interior_neighbor_offset": np.array([1, 0, 0])
        }
    }
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=True)
    out = capfd.readouterr().err
    assert "WARNING: No cells for boundary" in out


def test_dirichlet_velocity_only():
    v, p = dummy_fields()
    mesh = dummy_mesh()
    idx = np.array([[0, 1, 1]])
    mesh["boundary_conditions"] = {
        "inlet": {
            "type": "dirichlet",
            "cell_indices": idx,
            "velocity": np.array([5.0, 0.0, 0.0]),
            "pressure": 999.0,
            "apply_to": ["velocity"],
            "boundary_dim": 0,
            "interior_neighbor_offset": np.array([1, 0, 0])
        }
    }
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=True)
    assert np.all(v[0, 1, 1] == [5.0, 0.0, 0.0])
    assert p[0, 1, 1] == 0.0  # Not applied during tentative step


def test_dirichlet_pressure_only_final_step():
    v, p = dummy_fields()
    mesh = dummy_mesh()
    idx = np.array([[0, 1, 1]])
    mesh["boundary_conditions"] = {
        "wall": {
            "type": "dirichlet",
            "cell_indices": idx,
            "velocity": np.zeros(3),
            "pressure": 1337.0,
            "apply_to": ["pressure"],
            "boundary_dim": 0,
            "interior_neighbor_offset": np.array([1, 0, 0])
        }
    }
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=False)
    assert p[0, 1, 1] == 1337.0


def test_neumann_velocity_mirroring():
    v, p = dummy_fields((4, 4, 4))
    v[1, 2, 2] = [2.0, 1.0, -1.0]  # interior neighbor
    mesh = dummy_mesh((4, 4, 4))
    mesh["boundary_conditions"] = {
        "outlet": {
            "type": "neumann",
            "cell_indices": np.array([[0, 2, 2]]),  # boundary cell
            "velocity": None,
            "pressure": None,
            "apply_to": ["velocity"],
            "boundary_dim": 0,
            "interior_neighbor_offset": np.array([1, 0, 0])
        }
    }
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=True)
    assert np.allclose(v[0, 2, 2], [2.0, 1.0, -1.0])


def test_pressure_outlet_applies_pressure_only_after_projection():
    v, p = dummy_fields()
    mesh = dummy_mesh()
    idx = np.array([[3, 3, 3]])
    mesh["boundary_conditions"] = {
        "exit": {
            "type": "pressure_outlet",
            "cell_indices": idx,
            "velocity": np.zeros(3),
            "pressure": 101325.0,
            "apply_to": ["pressure"],
            "boundary_dim": 0,
            "interior_neighbor_offset": np.array([-1, 0, 0])
        }
    }

    # Tentative step — should skip pressure
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=True)
    assert p[3, 3, 3] == 0.0

    # Final step — should apply
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=False)
    assert p[3, 3, 3] == 101325.0


def test_unknown_bc_type_warns(capfd):
    v, p = dummy_fields()
    mesh = dummy_mesh()
    idx = np.array([[1, 1, 1]])
    mesh["boundary_conditions"] = {
        "strange": {
            "type": "mythic",
            "cell_indices": idx,
            "velocity": np.ones(3),
            "pressure": 0.0,
            "apply_to": ["velocity", "pressure"],
            "boundary_dim": 0,
            "interior_neighbor_offset": np.array([-1, 0, 0])
        }
    }
    apply_boundary_conditions(v, p, {}, mesh, is_tentative_step=True)
    err = capfd.readouterr().err
    assert "Unknown BC type" in err



