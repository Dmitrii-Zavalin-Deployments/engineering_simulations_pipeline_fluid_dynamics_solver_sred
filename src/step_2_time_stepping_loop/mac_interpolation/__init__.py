# src/step_2_time_stepping_loop/mac_interpolation/__init__.py
# 📦 Public API for MAC Interpolation
# Re-export vx, vy, vz interpolation functions from split modules

from .vx import (
    vx_i_plus_half,
    vx_i_minus_half,
    vx_i_plus_three_half,
    vx_i_minus_three_half,
    vx_j_plus_one,
    vx_j_minus_one,
    vx_k_plus_one,
    vx_k_minus_one,
)

from .vy import (
    vy_j_plus_half,
    vy_j_minus_half,
    vy_j_plus_three_half,
    vy_j_minus_three_half,
    vy_i_plus_one,
    vy_i_minus_one,
    vy_k_plus_one,
    vy_k_minus_one,
)

from .vz import (
    vz_k_plus_half,
    vz_k_minus_half,
    vz_k_plus_three_half,
    vz_k_minus_three_half,
    vz_i_plus_one,
    vz_i_minus_one,
    vz_j_plus_one,
    vz_j_minus_one,
)

# Optional: expose helpers if you want them available outside
from .base import _resolve_timestep, _get_velocity

__all__ = [
    # vx
    "vx_i_plus_half",
    "vx_i_minus_half",
    "vx_i_plus_three_half",
    "vx_i_minus_three_half",
    "vx_j_plus_one",
    "vx_j_minus_one",
    "vx_k_plus_one",
    "vx_k_minus_one",
    # vy
    "vy_j_plus_half",
    "vy_j_minus_half",
    "vy_j_plus_three_half",
    "vy_j_minus_three_half",
    "vy_i_plus_one",
    "vy_i_minus_one",
    "vy_k_plus_one",
    "vy_k_minus_one",
    # vz
    "vz_k_plus_half",
    "vz_k_minus_half",
    "vz_k_plus_three_half",
    "vz_k_minus_three_half",
    "vz_i_plus_one",
    "vz_i_minus_one",
    "vz_j_plus_one",
    "vz_j_minus_one",
    # base helpers
    "_resolve_timestep",
    "_get_velocity",
]



