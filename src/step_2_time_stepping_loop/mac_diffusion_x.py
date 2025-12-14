# src/step_2_time_stepping_loop/mac_diffusion_x.py
from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_interpolation.vx import (
    vx_i_plus_half, vx_i_minus_half, vx_i_plus_three_half,
    vx_j_plus_one, vx_j_minus_one,
    vx_k_plus_one, vx_k_minus_one,
)

debug = False

def laplacian_vx(cell_dict: Dict[str, Any], center: int,
                 dx: float, dy: float, dz: float,
                 timestep: int | None = None) -> float:
    """
    Compute full ∇²vx at a cell using MAC face velocities.
    Includes contributions from x-, y-, and z-directions:

        ∇²vx = ∂²vx/∂x² + ∂²vx/∂y² + ∂²vx/∂z²

    Stencils:
      - ∂²vx/∂x²: (vx(i+3/2) - 2*vx(i+1/2) + vx(i-1/2)) / dx^2
      - ∂²vx/∂y²: (vx(j+1) - 2*vx(j) + vx(j-1)) / dy^2
      - ∂²vx/∂z²: (vx(k+1) - 2*vx(k) + vx(k-1)) / dz^2

    Ghost-cell handling: interpolation functions enforce Neumann fallback if neighbors are missing.
    """
    # ∂²vx/∂x²
    v_ip3half = vx_i_plus_three_half(cell_dict, center, timestep)
    v_iphalf = vx_i_plus_half(cell_dict, center, timestep)
    v_iminushalf = vx_i_minus_half(cell_dict, center, timestep)
    d2vx_dx2 = (v_ip3half - 2.0 * v_iphalf + v_iminushalf) / (dx * dx)

    # ∂²vx/∂y²
    vx_jplus = vx_j_plus_one(cell_dict, center, timestep)
    vx_j = vx_i_plus_half(cell_dict, center, timestep)  # current vx at (i+1/2, j, k)
    vx_jminus = vx_j_minus_one(cell_dict, center, timestep)
    d2vx_dy2 = (vx_jplus - 2.0 * vx_j + vx_jminus) / (dy * dy)

    # ∂²vx/∂z²
    vx_kplus = vx_k_plus_one(cell_dict, center, timestep)
    vx_k = vx_i_plus_half(cell_dict, center, timestep)  # current vx at (i+1/2, j, k)
    vx_kminus = vx_k_minus_one(cell_dict, center, timestep)
    d2vx_dz2 = (vx_kplus - 2.0 * vx_k + vx_kminus) / (dz * dz)

    out = d2vx_dx2 + d2vx_dy2 + d2vx_dz2
    if debug:
        print(f"∇²vx at cell {center}: dx2={d2vx_dx2}, dy2={d2vx_dy2}, dz2={d2vx_dz2}, result={out}")
    return out



