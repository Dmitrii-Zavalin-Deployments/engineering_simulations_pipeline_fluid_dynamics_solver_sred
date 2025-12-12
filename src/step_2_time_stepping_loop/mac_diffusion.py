# src/step_2_time_stepping_loop/mac_diffusion.py
from typing import Dict, Any
from src.step_2_time_stepping_loop.mac_diffusion_x import laplacian_vx
from src.step_2_time_stepping_loop.mac_diffusion_y import laplacian_vy
from src.step_2_time_stepping_loop.mac_diffusion_z import laplacian_vz

debug = False  # toggle for verbose logging

def laplacian_velocity(cell_dict: Dict[str, Any], center: int,
                       dx: float, dy: float, dz: float,
                       timestep: int | None = None) -> Dict[str, float]:
    """
    Compute vector Laplacian ∇²v = (∇²vx, ∇²vy, ∇²vz) at a central cell.
    """
    out = {
        "vx": laplacian_vx(cell_dict, center, dx, dy, dz, timestep),
        "vy": laplacian_vy(cell_dict, center, dx, dy, dz, timestep),
        "vz": laplacian_vz(cell_dict, center, dx, dy, dz, timestep),
    }
    if debug:
        print(f"∇²v at cell {center}: {out}")
    return out



