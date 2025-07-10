# src/grid_modules/cell.py

from dataclasses import dataclass

@dataclass
class Cell:
    x: float
    y: float
    z: float
    velocity: list  # [vx, vy, vz]
    pressure: float



