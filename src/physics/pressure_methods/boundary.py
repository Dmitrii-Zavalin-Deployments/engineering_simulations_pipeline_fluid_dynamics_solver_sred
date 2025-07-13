# src/physics/pressure_methods/boundary.py
# 🧱 Boundary condition enforcement for pressure solver — ghost-aware and solid-safe

from typing import Tuple, List, Dict, Set

def apply_neumann_conditions(coord: Tuple[float, float, float],
                             neighbor: Tuple[float, float, float],
                             pressure_map: Dict[Tuple[float, float, float], float]) -> float:
    """
    Apply Neumann condition (zero-gradient) for missing or non-fluid neighbor.

    Args:
        coord: Current cell coordinate
        neighbor: Neighbor coordinate
        pressure_map: Pressure values

    Returns:
        Approximated pressure value using Neumann boundary logic
    """
    # Neumann: assume pressure gradient is zero ⇒ neighbor has same pressure
    return pressure_map.get(coord, 0.0)


def handle_solid_or_ghost_neighbors(coord: Tuple[float, float, float],
                                    neighbors: List[Tuple[float, float, float]],
                                    pressure_map: Dict[Tuple[float, float, float], float],
                                    fluid_mask_map: Dict[Tuple[float, float, float], bool],
                                    ghost_coords: Set[Tuple[float, float, float]]) -> float:
    """
    Adjust pressure update for neighbors that may be solid or ghost.

    Args:
        coord: Current cell coordinate
        neighbors: Neighbor coordinates (six directions)
        pressure_map: Known pressure values
        fluid_mask_map: Map of fluid vs solid states
        ghost_coords: Set of coordinates for ghost cells

    Returns:
        Sum of neighbor pressures with fallback handling for non-fluid neighbors
    """
    total = 0.0
    for n in neighbors:
        if n in ghost_coords:
            # Ghost neighbor: treat as Neumann boundary
            total += apply_neumann_conditions(coord, n, pressure_map)
        elif n in fluid_mask_map:
            if fluid_mask_map[n]:
                total += pressure_map.get(n, pressure_map.get(coord, 0.0))
            else:
                # Solid neighbor: Neumann fallback
                total += apply_neumann_conditions(coord, n, pressure_map)
        else:
            # Outside domain or missing neighbor: Neumann fallback
            total += apply_neumann_conditions(coord, n, pressure_map)
    return total



