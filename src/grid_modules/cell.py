# src/grid_modules/cell.py

from dataclasses import dataclass
from typing import List

@dataclass
class Cell:
    x: float
    y: float
    z: float
    velocity: List[float]  # [vx, vy, vz]
    pressure: float



