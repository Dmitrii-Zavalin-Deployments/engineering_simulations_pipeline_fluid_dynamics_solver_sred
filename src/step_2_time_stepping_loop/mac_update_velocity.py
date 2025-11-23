# src/step_2_time_stepping_loop/mac_update_velocity.py
# 🚀 MAC Velocity Update — Phase 1 predictor step for Navier–Stokes
#
# Implements velocity prediction:
#   v_* = v^n + (Δt/ρ)[ μ ∇²v^n − ρ Adv(v)^n − ∇p^n + F_face ]
#
# Inputs:
#   - cell_dict: grid dictionary with time_history
#   - center: flat index of central cell
#   - dx, dy, dz: grid spacings
#   - dt: timestep size
#   - rho: fluid density
#   - mu: fluid viscosity
#   - config: full input configuration (for external forces)
#   - timestep: current timestep index
#
# Outputs:
#   - v_x*, v_y*, v_z* at MAC faces

from typing import Dict, Any

from src.step_2_time_stepping_loop.mac_diffusion import laplacian_vx, laplacian_vy, laplacian_vz
from src.step_2_time_stepping_loop.mac_advection_ops import adv_vx, adv_vy, adv_vz
from src.step_2_time_stepping_loop.mac_gradients import grad_p_x, grad_p_y, grad_p_z
from src.step_2_time_stepping_loop.force_utils import load_external_forces
from src.step_2_time_stepping_loop.mac_interpolation.vx import vx_i_plus_half
from src.step_2_time_stepping_loop.mac_interpolation.vy import vy_j_plus_half
from src.step_2_time_stepping_loop.mac_interpolation.vz import vz_k_plus_half

debug = False  # toggle for verbose logging


def update_velocity_x(cell_dict: Dict[str, Any], center: int,
                      dx: float, dy: float, dz: float,
                      dt: float, rho: float, mu: float,
                      config: Dict[str, Any], timestep: int | None = None) -> float:
    """Predict intermediate v_x* at i+1/2 face."""
    v_n = vx_i_plus_half(cell_dict, center, timestep)
    lap = laplacian_vx(cell_dict, center, dx, timestep)   # FIXED
    adv = adv_vx(cell_dict, center, dx, dy, dz, timestep)
    gradp = grad_p_x(cell_dict, center, dx, timestep)     # FIXED
    forces = load_external_forces(config)
    F_face = forces["Fx"]

    v_star = v_n + (dt / rho) * (mu * lap - rho * adv - gradp + F_face)

    if debug:
        print(f"[Update vx] center={center}, v_n={v_n}, lap={lap}, adv={adv}, gradp={gradp}, F={F_face} -> v*={v_star}")

    return v_star


def update_velocity_y(cell_dict: Dict[str, Any], center: int,
                      dx: float, dy: float, dz: float,
                      dt: float, rho: float, mu: float,
                      config: Dict[str, Any], timestep: int | None = None) -> float:
    """Predict intermediate v_y* at j+1/2 face."""
    v_n = vy_j_plus_half(cell_dict, center, timestep)
    lap = laplacian_vy(cell_dict, center, dy, timestep)   # FIXED
    adv = adv_vy(cell_dict, center, dx, dy, dz, timestep)
    gradp = grad_p_y(cell_dict, center, dy, timestep)     # FIXED
    forces = load_external_forces(config)
    F_face = forces["Fy"]

    v_star = v_n + (dt / rho) * (mu * lap - rho * adv - gradp + F_face)

    if debug:
        print(f"[Update vy] center={center}, v_n={v_n}, lap={lap}, adv={adv}, gradp={gradp}, F={F_face} -> v*={v_star}")

    return v_star


def update_velocity_z(cell_dict: Dict[str, Any], center: int,
                      dx: float, dy: float, dz: float,
                      dt: float, rho: float, mu: float,
                      config: Dict[str, Any], timestep: int | None = None) -> float:
    """Predict intermediate v_z* at k+1/2 face."""
    v_n = vz_k_plus_half(cell_dict, center, timestep)
    lap = laplacian_vz(cell_dict, center, dz, timestep)   # FIXED
    adv = adv_vz(cell_dict, center, dx, dy, dz, timestep)
    gradp = grad_p_z(cell_dict, center, dz, timestep)     # FIXED
    forces = load_external_forces(config)
    F_face = forces["Fz"]

    v_star = v_n + (dt / rho) * (mu * lap - rho * adv - gradp + F_face)

    if debug:
        print(f"[Update vz] center={center}, v_n={v_n}, lap={lap}, adv={adv}, gradp={gradp}, F={F_face} -> v*={v_star}")

    return v_star



