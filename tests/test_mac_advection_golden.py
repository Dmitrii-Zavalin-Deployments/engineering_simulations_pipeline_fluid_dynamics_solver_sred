# tests/test_mac_advection_golden.py
# Golden tests: compute advection manually (using interpolation + stencils)
# and compare against adv_vx/adv_vy/adv_vz outputs for the central cell (13).

import pytest

from tests.mocks.cell_dict_mock import cell_dict

# Import the production advection operators
from src.step_2_time_stepping_loop.mac_advection import (
    adv_vx,
    adv_vy,
    adv_vz,
)

# Import the production interpolation functions for face sampling
from src.step_2_time_stepping_loop.mac_interpolation.vx import (
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
)
from src.step_2_time_stepping_loop.mac_interpolation.vy import (
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
)
from src.step_2_time_stepping_loop.mac_interpolation.vz import (
    vz_k_plus_half,
    vz_k_minus_half,
    vz_k_plus_three_half,
)


def _neighbor_index(cell_dict, center, key):
    """Helper to fetch a neighbor flat_index from the mock dictionary."""
    entry = cell_dict.get(str(center))
    if not entry:
        return None
    return entry.get(key)


# ---------------- Manual helpers (central differences at faces) ----------------
# These functions compute gradients of vx, vy, vz using the same interpolation
# functions as the production code, ensuring the manual calculation mirrors
# the implementation exactly.

def _manual_grad_vx_at_xface(cell_dict, center, dx, dy, dz, timestep):
    # ∂vx/∂x at the x-face using values at i+3/2 and i-1/2
    dvx_dx = (vx_i_plus_three_half(cell_dict, center, timestep) -
              vx_i_minus_half(cell_dict, center, timestep)) / (2.0 * dx)

    # ∂vx/∂y using j+1 and j-1 neighbors
    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vx_j_plus = vx_i_plus_half(cell_dict, j_plus, timestep) if j_plus else vx_i_plus_half(cell_dict, center, timestep)
    vx_j_minus = vx_i_plus_half(cell_dict, j_minus, timestep) if j_minus else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dy = (vx_j_plus - vx_j_minus) / (2.0 * dy)

    # ∂vx/∂z using k+1 and k-1 neighbors
    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vx_k_plus = vx_i_plus_half(cell_dict, k_plus, timestep) if k_plus else vx_i_plus_half(cell_dict, center, timestep)
    vx_k_minus = vx_i_plus_half(cell_dict, k_minus, timestep) if k_minus else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dz = (vx_k_plus - vx_k_minus) / (2.0 * dz)

    return dvx_dx, dvx_dy, dvx_dz


def _manual_grad_vy_at_yface(cell_dict, center, dx, dy, dz, timestep):
    # ∂vy/∂y at the y-face using j+3/2 and j-1/2
    dvy_dy = (vy_j_plus_three_half(cell_dict, center, timestep) -
              vy_j_minus_half(cell_dict, center, timestep)) / (2.0 * dy)

    # ∂vy/∂x using i+1 and i-1 neighbors
    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vy_i_plus = vy_j_plus_half(cell_dict, i_plus, timestep) if i_plus else vy_j_plus_half(cell_dict, center, timestep)
    vy_i_minus = vy_j_plus_half(cell_dict, i_minus, timestep) if i_minus else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dx = (vy_i_plus - vy_i_minus) / (2.0 * dx)

    # ∂vy/∂z using k+1 and k-1 neighbors
    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vy_k_plus = vy_j_plus_half(cell_dict, k_plus, timestep) if k_plus else vy_j_plus_half(cell_dict, center, timestep)
    vy_k_minus = vy_j_plus_half(cell_dict, k_minus, timestep) if k_minus else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dz = (vy_k_plus - vy_k_minus) / (2.0 * dz)

    return dvy_dx, dvy_dy, dvy_dz


def _manual_grad_vz_at_zface(cell_dict, center, dx, dy, dz, timestep):
    # ∂vz/∂z at the z-face using k+3/2 and k-1/2
    dvz_dz = (vz_k_plus_three_half(cell_dict, center, timestep) -
              vz_k_minus_half(cell_dict, center, timestep)) / (2.0 * dz)

    # ∂vz/∂x using i+1 and i-1 neighbors
    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vz_i_plus = vz_k_plus_half(cell_dict, i_plus, timestep) if i_plus else vz_k_plus_half(cell_dict, center, timestep)
    vz_i_minus = vz_k_plus_half(cell_dict, i_minus, timestep) if i_minus else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dx = (vz_i_plus - vz_i_minus) / (2.0 * dx)

    # ∂vz/∂y using j+1 and j-1 neighbors
    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vz_j_plus = vz_k_plus_half(cell_dict, j_plus, timestep) if j_plus else vz_k_plus_half(cell_dict, center, timestep)
    vz_j_minus = vz_k_plus_half(cell_dict, j_minus, timestep) if j_minus else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dy = (vz_j_plus - vz_j_minus) / (2.0 * dy)

    return dvz_dx, dvz_dy, dvz_dz


# ---------------- Golden tests for central cell (13) ----------------

def test_golden_adv_vx_central_t0():
    """
    Purpose:
      - Compute Adv(vx) at central cell 13, timestep 0, in two ways:
        1) Manually via interpolation + central differences.
        2) Using adv_vx.
      - Compare with tight tolerance to guarantee mathematical equivalence.

    How:
      - Collocate u, v, w at the x-face using production interpolation functions.
      - Compute ∂vx/∂x, ∂vx/∂y, ∂vx/∂z via central differences on the MAC grid.
      - Assemble u*∂vx/∂x + v*∂vx/∂y + w*∂vx/∂z.
      - Compare to adv_vx output.
    """
    dx = dy = dz = 1.0
    center = 13
    t = 0

    # Collocated face velocities
    u_face = vx_i_plus_half(cell_dict, center, t)
    v_face = vy_j_plus_half(cell_dict, center, t)
    w_face = vz_k_plus_half(cell_dict, center, t)

    # Manual gradients
    dvx_dx, dvx_dy, dvx_dz = _manual_grad_vx_at_xface(cell_dict, center, dx, dy, dz, t)

    manual = u_face * dvx_dx + v_face * dvx_dy + w_face * dvx_dz
    result = adv_vx(cell_dict, center, dx, dy, dz, timestep=t)

    assert pytest.approx(result, rel=1e-12, abs=1e-12) == manual


def test_golden_adv_vy_central_t0():
    """
    Purpose:
      - Golden test for Adv(vy) at central cell 13, timestep 0.
      - Ensures adv_vy matches the manual central-difference assembly.

    How:
      - Use vy face value for v, and collocate u, w to the y-face.
      - Compute ∂vy/∂x, ∂vy/∂y, ∂vy/∂z manually via interpolation + neighbors.
      - Compare to adv_vy output.
    """
    dx = dy = dz = 1.0
    center = 13
    t = 0

    # Collocated face velocities
    v_face = vy_j_plus_half(cell_dict, center, t)
    u_face = vx_i_plus_half(cell_dict, center, t)
    w_face = vz_k_plus_half(cell_dict, center, t)

    # Manual gradients
    dvy_dx, dvy_dy, dvy_dz = _manual_grad_vy_at_yface(cell_dict, center, dx, dy, dz, t)

    manual = u_face * dvy_dx + v_face * dvy_dy + w_face * dvy_dz
    result = adv_vy(cell_dict, center, dx, dy, dz, timestep=t)

    assert pytest.approx(result, rel=1e-12, abs=1e-12) == manual


def test_golden_adv_vz_central_t0():
    """
    Purpose:
      - Golden test for Adv(vz) at central cell 13, timestep 0.
      - Ensures adv_vz matches the manual central-difference assembly.

    How:
      - Use wz face value for w, and collocate u, v to the z-face.
      - Compute ∂vz/∂x, ∂vz/∂y, ∂vz/∂z manually via interpolation + neighbors.
      - Compare to adv_vz output.
    """
    dx = dy = dz = 1.0
    center = 13
    t = 0

    # Collocated face velocities
    w_face = vz_k_plus_half(cell_dict, center, t)
    u_face = vx_i_plus_half(cell_dict, center, t)
    v_face = vy_j_plus_half(cell_dict, center, t)

    # Manual gradients
    dvz_dx, dvz_dy, dvz_dz = _manual_grad_vz_at_zface(cell_dict, center, dx, dy, dz, t)

    manual = u_face * dvz_dx + v_face * dvz_dy + w_face * dvz_dz
    result = adv_vz(cell_dict, center, dx, dy, dz, timestep=t)

    assert pytest.approx(result, rel=1e-12, abs=1e-12) == manual



