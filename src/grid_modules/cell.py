# src/grid_modules/cell.py
# 🧬 Cell Definition — encapsulates per-cell physical state for Navier-Stokes
# simulation
# 📌 This dataclass defines the simulation grid's atomic unit.
# It is used across solver, diagnostics, and export routines.
# The fluid_mask flag is the sole determinant for solver inclusion/exclusion.

from dataclasses import dataclass
from typing import List

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


@dataclass
class Cell:
    x: float                 # Physical x-coordinate
    y: float                 # Physical y-coordinate
    z: float                 # Physical z-coordinate
    velocity: List[float]    # Velocity vector [vx, vy, vz]
    pressure: float          # Scalar pressure value
    fluid_mask: bool         # ✅ True if cell contains fluid, False if solid
