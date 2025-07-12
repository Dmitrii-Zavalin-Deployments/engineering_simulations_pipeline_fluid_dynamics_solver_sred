# src/physics/advection_methods/helpers.py
# 🔧 Math support utilities and cell manipulation

from src.grid_modules.cell import Cell
from typing import Optional, List

def copy_cell(
    cell: Cell,
    velocity: Optional[List[float]] = None,
    pressure: Optional[float] = None
) -> Cell:
    """
    Returns a new Cell object with optional velocity or pressure override.

    Args:
        cell (Cell): Original cell
        velocity (Optional[List[float]]): New velocity vector
        pressure (Optional[float]): New pressure value

    Returns:
        Cell: Copied cell with updated fields if specified
    """
    return Cell(
        x=cell.x,
        y=cell.y,
        z=cell.z,
        velocity=velocity if velocity is not None else cell.velocity,
        pressure=pressure if pressure is not None else cell.pressure,
        fluid_mask=cell.fluid_mask
    )



