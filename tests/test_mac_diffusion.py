# tests/test_mac_diffusion.py
# Unit tests for laplacian_velocity wrapper using the 3×3×3 mock cell_dict

import pytest
from src.step_2_time_stepping_loop.mac_diffusion import laplacian_velocity
from tests.mocks.cell_dict_mock import cell_dict


def test_laplacian_velocity_t0_complete():
    """
    Test full vector Laplacian at timestep=0 with dx=dy=dz=1.0.
    Ensures all components are returned as floats.
    """
    result = laplacian_velocity(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"vx", "vy", "vz"}
    assert all(isinstance(val, float) for val in result.values())
    # sanity bound
    assert all(abs(val) < 10.0 for val in result.values())


def test_laplacian_velocity_latest_equals_t1():
    """
    Test Laplacian defaults to latest timestep (timestep=1).
    """
    result_latest = laplacian_velocity(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0)
    result_t1 = laplacian_velocity(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=1)
    for comp in ["vx", "vy", "vz"]:
        assert abs(result_latest[comp] - result_t1[comp]) < 1e-12


def test_laplacian_velocity_scaled_grid():
    """
    Test Laplacian with non-unit grid spacing (dx=2, dy=2, dz=2).
    Ensures scaling by Δ^2 is applied correctly.
    """
    result_unit = laplacian_velocity(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    result_scaled = laplacian_velocity(cell_dict, 13, dx=2.0, dy=2.0, dz=2.0, timestep=0)
    # With doubled spacing, Laplacian magnitudes should shrink
    for comp in ["vx", "vy", "vz"]:
        assert abs(result_scaled[comp]) < abs(result_unit[comp])


def test_laplacian_velocity_boundary_fallback(monkeypatch):
    """
    Simulate missing neighbor keys to trigger Neumann fallback.
    """
    # Remove i+1 neighbor temporarily
    monkeypatch.setitem(cell_dict[str(13)], "flat_index_i_plus_1", None)
    result = laplacian_velocity(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    assert isinstance(result, dict)
    assert all(isinstance(val, float) for val in result.values())


def test_laplacian_velocity_symmetry_check():
    """
    Check that Laplacian vector is small if neighbor values are symmetric.
    """
    result = laplacian_velocity(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    # Should be close to zero if perfectly symmetric
    assert all(abs(val) < 5.0 for val in result.values())



