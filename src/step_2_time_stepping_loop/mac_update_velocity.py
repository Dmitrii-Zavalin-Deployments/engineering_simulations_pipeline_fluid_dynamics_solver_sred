# src/step_2_time_stepping_loop/mac_update_velocity.py
# 🚀 MAC Velocity Update — Phase 1 predictor step for Navier–Stokes
#
# Implements velocity prediction:
#   v_* = v^n + (Δt/ρ)[ μ ∇²v^n − ρ Adv(v)^n − ∇p^n + F_face ]
#
# Inputs:
#   - cell_dict: grid dictionary with time_history
#   - center: flat index of central cell
#   - config: full input configuration (JSON dict)
#   - timestep: current timestep index (None -> latest available)
#
# Outputs:
#   - v_x*, v_y*, v_z* at MAC faces (predictor velocities)
#
# Notes:
#   - Boundary handling: underlying interpolation/gradient operators apply Neumann fallbacks.
#   - Integration: predictor velocities should be staged (e.g., under "{timestep+1}_predictor")
#     and NOT committed as final until PPE (Phase 2) and velocity correction (Phase 3) are applied.

from typing import Dict, Any

from src.step_2_time_stepping_loop.mac_diffusion import laplacian_vx, laplacian_vy, laplacian_vz
from src.step_2_time_stepping_loop.mac_advection_ops import adv_vx, adv_vy, adv_vz
from src.step_2_time_stepping_loop.mac_gradients import grad_p_x, grad_p_y, grad_p_z
from src.step_2_time_stepping_loop.mac_interpolation.vx import vx_i_plus_half
from src.step_2_time_stepping_loop.mac_interpolation.vy import vy_j_plus_half
from src.step_2_time_stepping_loop.mac_interpolation.vz import vz_k_plus_half
from src.step_2_time_stepping_loop.parameter_utils import load_solver_parameters

debug = False  # toggle for verbose logging


def update_velocity_x(cell_dict: Dict[str, Any], center: int,
                      config: Dict[str, Any], timestep: int | None = None) -> float:
    """Predict intermediate v_x* at i+1/2 face."""
    params = load_solver_parameters(config)
    v_n: float = vx_i_plus_half(cell_dict, center, timestep)
    lap: float = laplacian_vx(cell_dict, center, params["dx"], timestep)
    adv: float = adv_vx(cell_dict, center, params["dx"], params["dy"], params["dz"], timestep)
    gradp: float = grad_p_x(cell_dict, center, params["dx"], timestep)

    v_star: float = v_n + (params["dt"] / params["rho"]) * (
        params["mu"] * lap - params["rho"] * adv - gradp + params["Fx"]
    )

    if debug:
        print(f"[Update vx] center={center}, v_n={v_n}, lap={lap}, adv={adv}, gradp={gradp}, Fx={params['Fx']} -> v*={v_star}")

    return v_star


def update_velocity_y(cell_dict: Dict[str, Any], center: int,
                      config: Dict[str, Any], timestep: int | None = None) -> float:
    """Predict intermediate v_y* at j+1/2 face."""
    params = load_solver_parameters(config)
    v_n: float = vy_j_plus_half(cell_dict, center, timestep)
    lap: float = laplacian_vy(cell_dict, center, params["dy"], timestep)
    adv: float = adv_vy(cell_dict, center, params["dx"], params["dy"], params["dz"], timestep)
    gradp: float = grad_p_y(cell_dict, center, params["dy"], timestep)

    v_star: float = v_n + (params["dt"] / params["rho"]) * (
        params["mu"] * lap - params["rho"] * adv - gradp + params["Fy"]
    )

    if debug:
        print(f"[Update vy] center={center}, v_n={v_n}, lap={lap}, adv={adv}, gradp={gradp}, Fy={params['Fy']} -> v*={v_star}")

    return v_star


def update_velocity_z(cell_dict: Dict[str, Any], center: int,
                      config: Dict[str, Any], timestep: int | None = None) -> float:
    """Predict intermediate v_z* at k+1/2 face."""
    params = load_solver_parameters(config)
    v_n: float = vz_k_plus_half(cell_dict, center, timestep)
    lap: float = laplacian_vz(cell_dict, center, params["dz"], timestep)
    adv: float = adv_vz(cell_dict, center, params["dx"], params["dy"], params["dz"], timestep)
    gradp: float = grad_p_z(cell_dict, center, params["dz"], timestep)

    v_star: float = v_n + (params["dt"] / params["rho"]) * (
        params["mu"] * lap - params["rho"] * adv - gradp + params["Fz"]
    )

    if debug:
        print(f"[Update vz] center={center}, v_n={v_n}, lap={lap}, adv={adv}, gradp={gradp}, Fz={params['Fz']} -> v*={v_star}")

    return v_star



