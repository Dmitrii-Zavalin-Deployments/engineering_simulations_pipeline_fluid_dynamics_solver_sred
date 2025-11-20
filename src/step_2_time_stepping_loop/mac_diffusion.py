# src/step_2_time_stepping_loop/mac_diffusion.py
# 🌀 Step 2: MAC Diffusion — Compute ∇² operators (second-order derivatives) using v±3/2 stencils

from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_interpolation import (
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
    vz_k_plus_half,
    vz_k_minus_half,
    vz_k_plus_three_half,
)

debug = False  # toggle for verbose logging


# ---------------- Laplacian of Velocity Components ----------------

def laplacian_vx(cell_dict: Dict[str, Any], i_cell: int, dx: float, timestep: int | None = None) -> float:
    """
    Compute ∇²vx at a cell using MAC face velocities in x-direction.
    Formula: (vx(i+3/2) - 2*vx(i+1/2) + vx(i-1/2)) / dx^2

    Ghost-cell handling: interpolation functions already degrade gracefully if neighbors are missing.
    """
    v_ip3half = vx_i_plus_three_half(cell_dict, i_cell, timestep)
    v_iphalf = vx_i_plus_half(cell_dict, i_cell, timestep)
    v_iminushalf = vx_i_minus_half(cell_dict, i_cell, timestep)

    out = (v_ip3half - 2.0 * v_iphalf + v_iminushalf) / (dx * dx)
    if debug:
        print(f"∇²vx at cell {i_cell}: v+3/2={v_ip3half}, v+1/2={v_iphalf}, v-1/2={v_iminushalf}, result={out}")
    return out


def laplacian_vy(cell_dict: Dict[str, Any], j_cell: int, dy: float, timestep: int | None = None) -> float:
    """
    Compute ∇²vy at a cell using MAC face velocities in y-direction.
    Formula: (vy(j+3/2) - 2*vy(j+1/2) + vy(j-1/2)) / dy^2

    Ghost-cell handling: interpolation functions already degrade gracefully if neighbors are missing.
    """
    v_jp3half = vy_j_plus_three_half(cell_dict, j_cell, timestep)
    v_jphalf = vy_j_plus_half(cell_dict, j_cell, timestep)
    v_jminushalf = vy_j_minus_half(cell_dict, j_cell, timestep)

    out = (v_jp3half - 2.0 * v_jphalf + v_jminushalf) / (dy * dy)
    if debug:
        print(f"∇²vy at cell {j_cell}: v+3/2={v_jp3half}, v+1/2={v_jphalf}, v-1/2={v_jminushalf}, result={out}")
    return out


def laplacian_vz(cell_dict: Dict[str, Any], k_cell: int, dz: float, timestep: int | None = None) -> float:
    """
    Compute ∇²vz at a cell using MAC face velocities in z-direction.
    Formula: (vz(k+3/2) - 2*vz(k+1/2) + vz(k-1/2)) / dz^2

    Ghost-cell handling: interpolation functions already degrade gracefully if neighbors are missing.
    """
    v_kp3half = vz_k_plus_three_half(cell_dict, k_cell, timestep)
    v_kphalf = vz_k_plus_half(cell_dict, k_cell, timestep)
    v_kminushalf = vz_k_minus_half(cell_dict, k_cell, timestep)

    out = (v_kp3half - 2.0 * v_kphalf + v_kminushalf) / (dz * dz)
    if debug:
        print(f"∇²vz at cell {k_cell}: v+3/2={v_kp3half}, v+1/2={v_kphalf}, v-1/2={v_kminushalf}, result={out}")
    return out


# ---------------- Vector Laplacian ----------------

def laplacian_velocity(cell_dict: Dict[str, Any], center: int, dx: float, dy: float, dz: float, timestep: int | None = None) -> Dict[str, float]:
    """
    Compute vector Laplacian ∇²v = (∇²vx, ∇²vy, ∇²vz) at a central cell.
    Returns a dict with keys {"vx", "vy", "vz"}.
    """
    lx = laplacian_vx(cell_dict, center, dx, timestep)
    ly = laplacian_vy(cell_dict, center, dy, timestep)
    lz = laplacian_vz(cell_dict, center, dz, timestep)

    out = {"vx": lx, "vy": ly, "vz": lz}
    if debug:
        print(f"∇²v at cell {center}: {out}")
    return out



