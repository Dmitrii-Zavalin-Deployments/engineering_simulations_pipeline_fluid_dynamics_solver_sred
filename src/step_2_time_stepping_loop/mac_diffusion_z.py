# src/step_2_time_stepping_loop/mac_diffusion_z.py
from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_interpolation import (
    vz_k_plus_half, vz_k_minus_half, vz_k_plus_three_half,
    # TODO: vz_i±1, vz_j±1
)

debug = False  # toggle for verbose logging

def laplacian_vz(cell_dict: Dict[str, Any], center: int,
                 dx: float, dy: float, dz: float,
                 timestep: int | None = None) -> float:
    """
    Compute full ∇²vz at a cell using MAC face velocities.
    Includes contributions from x-, y-, and z-directions:

        ∇²vz = ∂²vz/∂x² + ∂²vz/∂y² + ∂²vz/∂z²

    Stencils:
      - ∂²vz/∂z²: (vz(k+3/2) - 2*vz(k+1/2) + vz(k-1/2)) / dz^2
      - ∂²vz/∂x²: (vz(i+1) - 2*vz(i) + vz(i-1)) / dx^2   [TODO: needs interpolation funcs]
      - ∂²vz/∂y²: (vz(j+1) - 2*vz(j) + vz(j-1)) / dy^2   [TODO: needs interpolation funcs]

    Ghost-cell handling: interpolation functions should degrade gracefully if neighbors are missing.
    """
    # ∂²vz/∂z²
    v_kp3half = vz_k_plus_three_half(cell_dict, center, timestep)
    v_kphalf = vz_k_plus_half(cell_dict, center, timestep)
    v_kminushalf = vz_k_minus_half(cell_dict, center, timestep)
    d2vz_dz2 = (v_kp3half - 2.0 * v_kphalf + v_kminushalf) / (dz * dz)

    # ∂²vz/∂x² (placeholder)
    d2vz_dx2 = 0.0

    # ∂²vz/∂y² (placeholder)
    d2vz_dy2 = 0.0

    out = d2vz_dx2 + d2vz_dy2 + d2vz_dz2
    if debug:
        print(f"∇²vz at cell {center}: dx2={d2vz_dx2}, dy2={d2vz_dy2}, dz2={d2vz_dz2}, result={out}")
    return out



