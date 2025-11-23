# src/step_2_time_stepping_loop/mac_advection_gradients.py
# 🌀 MAC Advection — gradient helpers for nonlinear convective terms
# Adv(v) = u ∂v/∂x + v ∂v/∂y + w ∂v/∂z, evaluated at the corresponding MAC face
#
# Boundary Fallback Theory in CFD:
# In finite-difference and finite-volume CFD solvers, advection terms are computed using neighboring cell values.
# At boundaries, however, a neighbor cell may not exist (e.g., outside the domain). If the solver attempts to access
# a missing neighbor, it risks a KeyError or undefined behavior.
#
# To prevent this, a common strategy is to reuse the central cell’s own velocity value when a neighbor is missing.
# This ensures the computation remains finite and stable.
#
# Why Reusing the Central Cell Works:
# Consider a central difference stencil for a velocity component v_x:
#
#   ∂v_x/∂y |_(i+1/2,j,k) ≈ ( v_x(i+1/2, j+1, k) - v_x(i+1/2, j-1, k) ) / (2 Δy)
#
# Normally, this requires both j+1 and j-1 neighbors.
# At a boundary, one of these neighbors may not exist.
# If we reuse the central cell’s value in place of the missing neighbor, the numerator collapses:
#
#   v_x(central) - v_x(central) = 0
#
# Thus, the derivative becomes zero.
#
# Link to Neumann Boundary Condition:
# This fallback enforces a zero-gradient boundary condition, also known as a Neumann condition:
#
#   ∂v/∂n = 0   at the boundary
#
# Physical meaning: No flux across the boundary; the velocity does not change in the normal direction.
# Numerical meaning: The solver avoids undefined values and maintains stability.
# Default behavior: In many CFD codes, when ghost cells are not explicitly defined, the Neumann condition is the safe default.

from typing import Dict, Any, Optional

# Import face interpolation functions
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

debug = False  # toggle for verbose logging

# ---------------- Utilities ----------------

def _neighbor_index(cell_dict: Dict[str, Any], center: int, key: str) -> Optional[int]:
    """Return neighbor flat_index via key (e.g., 'flat_index_j_plus_1') or None if missing."""
    d = cell_dict.get(str(center))
    if not d:
        return None
    return d.get(key, None)

# ---------------- Gradient helpers ----------------

def _grad_vx_at_xface(cell_dict, center, dx, dy, dz, timestep=None):
    """Gradients of v_x at the x-face (i+1/2)."""
    dvx_dx = (vx_i_plus_three_half(cell_dict, center, timestep) -
              vx_i_minus_half(cell_dict, center, timestep)) / (2.0 * dx)

    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vx_j_plus = vx_i_plus_half(cell_dict, j_plus, timestep) if j_plus else vx_i_plus_half(cell_dict, center, timestep)
    vx_j_minus = vx_i_plus_half(cell_dict, j_minus, timestep) if j_minus else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dy = (vx_j_plus - vx_j_minus) / (2.0 * dy)

    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vx_k_plus = vx_i_plus_half(cell_dict, k_plus, timestep) if k_plus else vx_i_plus_half(cell_dict, center, timestep)
    vx_k_minus = vx_i_plus_half(cell_dict, k_minus, timestep) if k_minus else vx_i_plus_half(cell_dict, center, timestep)
    dvx_dz = (vx_k_plus - vx_k_minus) / (2.0 * dz)

    return {"dx": dvx_dx, "dy": dvx_dy, "dz": dvx_dz}


def _grad_vy_at_yface(cell_dict, center, dx, dy, dz, timestep=None):
    """Gradients of v_y at the y-face (j+1/2)."""
    dvy_dy = (vy_j_plus_three_half(cell_dict, center, timestep) -
              vy_j_minus_half(cell_dict, center, timestep)) / (2.0 * dy)

    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vy_i_plus = vy_j_plus_half(cell_dict, i_plus, timestep) if i_plus else vy_j_plus_half(cell_dict, center, timestep)
    vy_i_minus = vy_j_plus_half(cell_dict, i_minus, timestep) if i_minus else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dx = (vy_i_plus - vy_i_minus) / (2.0 * dx)

    k_plus = _neighbor_index(cell_dict, center, "flat_index_k_plus_1")
    k_minus = _neighbor_index(cell_dict, center, "flat_index_k_minus_1")
    vy_k_plus = vy_j_plus_half(cell_dict, k_plus, timestep) if k_plus else vy_j_plus_half(cell_dict, center, timestep)
    vy_k_minus = vy_j_plus_half(cell_dict, k_minus, timestep) if k_minus else vy_j_plus_half(cell_dict, center, timestep)
    dvy_dz = (vy_k_plus - vy_k_minus) / (2.0 * dz)

    return {"dx": dvy_dx, "dy": dvy_dy, "dz": dvy_dz}


def _grad_vz_at_zface(cell_dict, center, dx, dy, dz, timestep=None):
    """Gradients of v_z at the z-face (k+1/2)."""
    dvz_dz = (vz_k_plus_three_half(cell_dict, center, timestep) -
              vz_k_minus_half(cell_dict, center, timestep)) / (2.0 * dz)

    i_plus = _neighbor_index(cell_dict, center, "flat_index_i_plus_1")
    i_minus = _neighbor_index(cell_dict, center, "flat_index_i_minus_1")
    vz_i_plus = vz_k_plus_half(cell_dict, i_plus, timestep) if i_plus else vz_k_plus_half(cell_dict, center, timestep)
    vz_i_minus = vz_k_plus_half(cell_dict, i_minus, timestep) if i_minus else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dx = (vz_i_plus - vz_i_minus) / (2.0 * dx)

    j_plus = _neighbor_index(cell_dict, center, "flat_index_j_plus_1")
    j_minus = _neighbor_index(cell_dict, center, "flat_index_j_minus_1")
    vz_j_plus = vz_k_plus_half(cell_dict, j_plus, timestep) if j_plus else vz_k_plus_half(cell_dict, center, timestep)
    vz_j_minus = vz_k_plus_half(cell_dict, j_minus, timestep) if j_minus else vz_k_plus_half(cell_dict, center, timestep)
    dvz_dy = (vz_j_plus - vz_j_minus) / (2.0 * dy)

    return {"dx": dvz_dx, "dy": dvz_dy, "dz": dvz_dz}



