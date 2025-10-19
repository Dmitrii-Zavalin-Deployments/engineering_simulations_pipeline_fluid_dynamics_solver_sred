# src/physics/pressure_methods/boundary.py
# 🧱 Boundary Condition Enforcement — ghost-aware and solid-safe pressure neighbor logic

from typing import Tuple, List, Dict, Set

def apply_neumann_conditions(coord: Tuple[float, float, float],
                             neighbor: Tuple[float, float, float],
                             pressure_map: Dict[Tuple[float, float, float], float]) -> float:
    """
    Apply Neumann condition (zero-gradient) for missing or non-fluid neighbor.

    Roadmap Alignment:
    Boundary Enforcement:
    - Neumann: ∂P/∂n = 0 ⇒ pressure at neighbor equals pressure at current cell

    Purpose:
    - Preserve solver stability at domain edges
    - Avoid artificial pressure gradients near solids or missing neighbors
    - Support reflex traceability and mutation diagnostics

    Args:
        coord: Current cell coordinate
        neighbor: Neighbor coordinate
        pressure_map: Pressure values

    Returns:
        Approximated pressure value using Neumann boundary logic
    """
    return pressure_map.get(coord, 0.0)


def handle_solid_or_ghost_neighbors(coord: Tuple[float, float, float],
                                    neighbors: List[Tuple[float, float, float]],
                                    pressure_map: Dict[Tuple[float, float, float], float],
                                    fluid_mask_map: Dict[Tuple[float, float, float], bool],
                                    ghost_coords: Set[Tuple[float, float, float]],
                                    ghost_pressure_map: Dict[Tuple[float, float, float], float]) -> float:
    """
    Adjust pressure update for neighbors that may be solid, ghost, or outside domain.

    Roadmap Alignment:
    Boundary Enforcement:
    - Fluid neighbors → standard pressure coupling
    - Ghost neighbors → Dirichlet enforcement (explicit pressure)
    - Solid or missing → Neumann fallback (zero-gradient)

    Purpose:
    - Ensure physical fidelity at boundaries
    - Modularize pressure logic for ghost-aware solvers
    - Support reflex diagnostics and mutation traceability

    Strategy:
    - Ghost cells: use explicit pressure if available, fallback to Neumann if missing
    - Solid cells: always fallback to Neumann
    - Missing/out-of-domain: fallback to Neumann

    Args:
        coord: Current cell coordinate
        neighbors: Neighbor coordinates (six directions)
        pressure_map: Known pressure values for fluid cells
        fluid_mask_map: Map of fluid vs solid states
        ghost_coords: Set of coordinates for ghost cells
        ghost_pressure_map: Explicit ghost pressure values from enforced boundaries

    Returns:
        Sum of neighbor pressures with fallback handling for non-fluid neighbors
    """
    total = 0.0
    for n in neighbors:
        if n in ghost_coords:
            # ✅ Ghost neighbor: use Dirichlet ghost pressure if available
            if n in ghost_pressure_map:
                total += ghost_pressure_map[n]
            else:
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



