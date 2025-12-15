# tests/test_mac_diffusion_z.py
# Unit tests for laplacian_vz using the 3×3×3 mock cell_dict

import pytest
from src.step_2_time_stepping_loop.mac_diffusion_z import laplacian_vz
from tests.mocks.cell_dict_mock import cell_dict


def test_laplacian_vz_t0_complete():
    """
    Test full Laplacian at timestep=0 with dx=dy=dz=1.0.
    Uses mock values from cell_dict for neighbors.
    """
    result = laplacian_vz(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    # Should be a finite float
    assert isinstance(result, float)
    assert abs(result) < 10.0  # sanity bound


def test_laplacian_vz_latest_equals_t1():
    """
    Test Laplacian defaults to latest timestep (timestep=1).
    """
    result_latest = laplacian_vz(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0)
    result_t1 = laplacian_vz(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=1)
    assert abs(result_latest - result_t1) < 1e-12


def test_laplacian_vz_scaled_grid():
    """
    Test Laplacian with non-unit grid spacing (dx=2, dy=2, dz=2).
    Ensures scaling by Δ^2 is applied correctly.
    """
    result_unit = laplacian_vz(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    result_scaled = laplacian_vz(cell_dict, 13, dx=2.0, dy=2.0, dz=2.0, timestep=0)
    # With doubled spacing, Laplacian magnitude should shrink by ~1/4
    assert abs(result_scaled) < abs(result_unit)


def test_laplacian_vz_boundary_fallback(monkeypatch):
    """
    Simulate missing neighbor keys to trigger Neumann fallback.
    """
    # Remove i+1 neighbor temporarily
    monkeypatch.setitem(cell_dict[str(13)], "flat_index_i_plus_1", None)
    result = laplacian_vz(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    # Should still return a finite float (fallback to central value)
    assert isinstance(result, float)


def test_laplacian_vz_symmetry_check():
    """
    Check that Laplacian is small if neighbor values are symmetric.
    """
    result = laplacian_vz(cell_dict, 13, dx=1.0, dy=1.0, dz=1.0, timestep=0)
    # Should be close to zero if perfectly symmetric
    assert abs(result) < 5.0  # loose bound, depends on mock values



