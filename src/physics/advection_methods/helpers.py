# src/physics/advection_methods/helpers.py
# 🔧 Math support utilities — vector operations, interpolation, cell manipulation

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


def vector_add(a: List[float], b: List[float]) -> List[float]:
    """
    Adds two 3D vectors component-wise.

    Args:
        a (List[float]): First vector
        b (List[float]): Second vector

    Returns:
        List[float]: Resulting vector
    """
    return [a[i] + b[i] for i in range(3)]


def vector_scale(v: List[float], scalar: float) -> List[float]:
    """
    Scales a vector by a scalar.

    Args:
        v (List[float]): Input vector
        scalar (float): Scalar multiplier

    Returns:
        List[float]: Scaled vector
    """
    return [scalar * comp for comp in v]



