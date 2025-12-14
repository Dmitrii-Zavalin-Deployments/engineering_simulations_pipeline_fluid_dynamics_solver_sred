# src/step_2_time_stepping_loop/mac_diffusion_y.py
from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_interpolation.vy import (
    vy_j_plus_half, vy_j_minus_half, vy_j_plus_three_half,
    vy_i_plus_one, vy_i_minus_one,
    vy_k_plus_one, vy_k_minus_one,
)

debug = False  # toggle for verbose logging

def laplacian_vy(cell_dict: Dict[str, Any], center: int,
                 dx: float, dy: float, dz: float,
                 timestep: int | None = None) -> float:
    """
    Compute full ∇²vy at a cell using MAC face velocities.
    Includes contributions from x-, y-, and z-directions:

        ∇²vy = ∂²vy/∂x² + ∂²vy/∂y² + ∂²vy/∂z²

    Stencils:
      - ∂²vy/∂y²: (vy(j+3/2) - 2*vy(j+1/2) + vy(j-1/2)) / dy^2
      - ∂²vy/∂x²: (vy(i+1) - 2*vy(i) + vy(i-1)) / dx^2
      - ∂²vy/∂z²: (vy(k+1) - 2*vy(k) + vy(k-1)) / dz^2

    Ghost-cell handling: interpolation functions enforce Neumann fallback if neighbors are missing.
    """
    # ∂²vy/∂y²
    v_jp3half = vy_j_plus_three_half(cell_dict, center, timestep)
    v_jphalf = vy_j_plus_half(cell_dict, center, timestep)
    v_jminushalf = vy_j_minus_half(cell_dict, center, timestep)
    d2vy_dy2 = (v_jp3half - 2.0 * v_jphalf + v_jminushalf) / (dy * dy)

    # ∂²vy/∂x²
    vy_ip1 = vy_i_plus_one(cell_dict, center, timestep)
    vy_i = vy_j_plus_half(cell_dict, center, timestep)  # current vy at (i, j+1/2, k)
    vy_im1 = vy_i_minus_one(cell_dict, center, timestep)
    d2vy_dx2 = (vy_ip1 - 2.0 * vy_i + vy_im1) / (dx * dx)

    # ∂²vy/∂z²
    vy_kp1 = vy_k_plus_one(cell_dict, center, timestep)
    vy_k = vy_j_plus_half(cell_dict, center, timestep)  # current vy at (i, j+1/2, k)
    vy_km1 = vy_k_minus_one(cell_dict, center, timestep)
    d2vy_dz2 = (vy_kp1 - 2.0 * vy_k + vy_km1) / (dz * dz)

    out = d2vy_dx2 + d2vy_dy2 + d2vy_dz2
    if debug:
        print(f"∇²vy at cell {center}: dx2={d2vy_dx2}, dy2={d2vy_dy2}, dz2={d2vy_dz2}, result={out}")
    return out



