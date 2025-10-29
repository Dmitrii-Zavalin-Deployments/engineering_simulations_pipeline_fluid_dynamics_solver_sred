# src/physics/advection_methods/helpers.py
# 🔧 Advection Math Utilities — vector operations, interpolation, and cell manipulation for ∂u/∂t routines
# 📌 This module supports velocity advection and cell mutation logic.
# It preserves fluid_mask status for solver inclusion integrity.
# It does NOT exclude cells — masking logic is enforced upstream.

from src.grid_modules.cell import Cell
from typing import Optional, List

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

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
    new_cell = Cell(
        x=cell.x,
        y=cell.y,
        z=cell.z,
        velocity=velocity if velocity is not None else cell.velocity,
        pressure=pressure if pressure is not None else cell.pressure,
        fluid_mask=cell.fluid_mask
    )

    if debug:
        print(f"[HELPERS] Copied cell @ ({new_cell.x:.2f}, {new_cell.y:.2f}, {new_cell.z:.2f}) → velocity={new_cell.velocity}, pressure={new_cell.pressure}, fluid_mask={new_cell.fluid_mask}")

    return new_cell


def vector_add(a: List[float], b: List[float]) -> List[float]:
    """
    Adds two 3D vectors component-wise.

    Args:
        a (List[float]): First vector
        b (List[float]): Second vector

    Returns:
        List[float]: Resulting vector
    """
    result = [a[i] + b[i] for i in range(3)]
    if debug:
        print(f"[HELPERS] vector_add → {a} + {b} = {result}")
    return result


def vector_scale(v: List[float], scalar: float) -> List[float]:
    """
    Scales a vector by a scalar.

    Args:
        v (List[float]): Input vector
        scalar (float): Scalar multiplier

    Returns:
        List[float]: Scaled vector
    """
    result = [scalar * comp for comp in v]
    if debug:
        print(f"[HELPERS] vector_scale → {v} * {scalar} = {result}")
    return result



