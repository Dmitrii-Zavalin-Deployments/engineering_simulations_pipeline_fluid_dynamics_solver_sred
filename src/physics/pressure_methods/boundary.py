# src/physics/pressure_methods/boundary.py
# 🧱 Boundary condition enforcement for pressure solver

from typing import Tuple, List, Dict

def apply_neumann_conditions(coord: Tuple[float, float, float],
                             neighbor: Tuple[float, float, float],
                             pressure_map: Dict[Tuple[float, float, float], float]) -> float:
    """
    Apply Neumann condition (zero-gradient) for missing neighbor.

    Args:
        coord: Current cell coordinate
        neighbor: Neighbor coordinate
        pressure_map: Pressure values

    Returns:
        Approximated pressure value using Neumann boundary logic
    """
    # Neumann: assume pressure gradient is zero across boundary ⇒ neighbor has same pressure
    return pressure_map.get(coord, 0.0)


def handle_solid_neighbors(coord: Tuple[float, float, float],
                           neighbors: List[Tuple[float, float, float]],
                           pressure_map: Dict[Tuple[float, float, float], float],
                           fluid_mask_map: Dict[Tuple[float, float, float], bool]) -> float:
    """
    Adjust pressure update for solid neighbors. For each neighbor:
    - If neighbor is fluid, use its pressure
    - If solid or missing, apply Neumann condition

    Args:
        coord: Current cell coordinate
        neighbors: Neighbor coordinates (six offsets)
        pressure_map: Pressure values for known cells
        fluid_mask_map: True for fluid cells, False for solids

    Returns:
        Sum of neighbor pressures with boundary-aware fallback
    """
    result = 0.0
    for n in neighbors:
        if n in fluid_mask_map:
            if fluid_mask_map[n]:
                result += pressure_map.get(n, pressure_map.get(coord, 0.0))
            else:
                # Solid neighbor: Neumann boundary
                result += apply_neumann_conditions(coord, n, pressure_map)
        else:
            # Missing neighbor: outside grid ⇒ Neumann boundary
            result += apply_neumann_conditions(coord, n, pressure_map)
    return result



