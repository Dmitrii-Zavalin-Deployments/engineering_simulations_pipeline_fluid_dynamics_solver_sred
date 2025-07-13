# src/physics/divergence_methods/divergence_helpers.py
# 🧮 Divergence helper utilities for central difference scheme

from src.grid_modules.cell import Cell
from typing import List, Tuple, Optional, Dict

def get_neighbor_velocity(
    grid_lookup: Dict[Tuple[float, float, float], Cell],
    x: float,
    y: float,
    z: float,
    axis: str,
    sign: int,
    spacing: float
) -> Optional[List[float]]:
    """
    Retrieves velocity vector from neighbor cell along specified axis and direction.

    Args:
        grid_lookup (Dict): (x, y, z) → Cell mapping
        x, y, z (float): Origin cell coordinates
        axis (str): Axis ('x', 'y', or 'z')
        sign (int): Direction (+1 or -1)
        spacing (float): Grid spacing along that axis

    Returns:
        Optional[List[float]]: Velocity if neighbor is fluid and valid; None otherwise
    """
    offset = {
        'x': (spacing, 0.0, 0.0),
        'y': (0.0, spacing, 0.0),
        'z': (0.0, 0.0, spacing)
    }.get(axis, (0.0, 0.0, 0.0))

    dx, dy, dz = offset
    target = (x + sign * dx, y + sign * dy, z + sign * dz)
    neighbor = grid_lookup.get(target)

    if neighbor and neighbor.fluid_mask and isinstance(neighbor.velocity, list):
        return neighbor.velocity
    return None

def central_gradient(
    v_pos: Optional[List[float]],
    v_neg: Optional[List[float]],
    spacing: float,
    component: int
) -> float:
    """
    Computes central difference gradient for a velocity component.

    Args:
        v_pos (Optional[List[float]]): Velocity vector from positive direction
        v_neg (Optional[List[float]]): Velocity vector from negative direction
        spacing (float): Grid spacing along axis
        component (int): Index of component to differentiate (0:x, 1:y, 2:z)

    Returns:
        float: Gradient value ∂v/∂axis or zero if neighbors are missing
    """
    if v_pos is not None and v_neg is not None:
        return (v_pos[component] - v_neg[component]) / (2.0 * spacing)
    return 0.0



