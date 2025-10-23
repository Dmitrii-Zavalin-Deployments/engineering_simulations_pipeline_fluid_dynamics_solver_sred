# src/physics/divergence_methods/divergence_helpers.py
# 🧮 Divergence Helpers — velocity access and gradient computation for central-difference ∇ · u enforcement
# 📌 This module supports continuity diagnostics and projection validation.
# It excludes only neighbors explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or boundary proximity.

from src.grid_modules.cell import Cell
from typing import List, Tuple, Optional, Dict

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

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
        if debug:
            print(f"[HELPERS] Neighbor @ {target} → velocity = {neighbor.velocity}")
        return neighbor.velocity

    if debug:
        print(f"[HELPERS] Neighbor @ {target} skipped → fluid_mask={getattr(neighbor, 'fluid_mask', None)}")
    return None

def central_difference(
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
        gradient = (v_pos[component] - v_neg[component]) / (2.0 * spacing)
        if debug:
            print(f"[HELPERS] ∂v[{component}]/∂axis = ({v_pos[component]} - {v_neg[component]}) / (2 * {spacing}) = {gradient:.6f}")
        return gradient
    if debug:
        print(f"[HELPERS] Central difference skipped → missing neighbors")
    return 0.0



