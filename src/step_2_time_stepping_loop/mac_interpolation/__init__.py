# src/step_2_time_stepping_loop/mac_interpolation/__init__.py
# 📦 Public API for MAC Interpolation
# Re-export vx, vy, vz interpolation functions from split modules

from .vx import (
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
    vx_i_minus_three_half,
)

from .vy import (
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
    vy_j_minus_three_half,
)

from .vz import (
    vz_k_plus_half,
    vz_k_minus_half,
    vz_k_plus_three_half,
    vz_k_minus_three_half,
)

# Optional: expose helpers if you want them available outside
from .base import _resolve_timestep, _get_velocity

__all__ = [
    # vx
    "vx_i_plus_half",
    "vx_i_minus_half",
    "vx_i_plus_three_half",
    "vx_i_minus_three_half",
    # vy
    "vy_j_plus_half",
    "vy_j_minus_half",
    "vy_j_plus_three_half",
    "vy_j_minus_three_half",
    # vz
    "vz_k_plus_half",
    "vz_k_minus_half",
    "vz_k_plus_three_half",
    "vz_k_minus_three_half",
    # base helpers
    "_resolve_timestep",
    "_get_velocity",
]



