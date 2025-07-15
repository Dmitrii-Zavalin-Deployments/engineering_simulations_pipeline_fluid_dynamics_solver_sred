# src/grid_modules/cell.py

from dataclasses import dataclass
from typing import List
import logging

@dataclass
class Cell:
    x: float
    y: float
    z: float
    velocity: List[float]  # [vx, vy, vz]
    pressure: float
    fluid_mask: bool       # ✅ True if cell contains fluid, False if solid

    def __post_init__(self):
        # 🧱 Graceful fallback: invalid velocity vectors downgrade the cell to solid for safety
        if not isinstance(self.velocity, list) or len(self.velocity) != 3:
            logging.debug(f"💡 [Cell fallback] Downgrading malformed velocity: {self}")
            self.velocity = None
            self.pressure = None
            self.fluid_mask = False



