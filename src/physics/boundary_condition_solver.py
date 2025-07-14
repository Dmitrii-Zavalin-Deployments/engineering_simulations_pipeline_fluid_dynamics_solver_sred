# src/physics/boundary_condition_solver.py
# 🧪 Boundary Condition Solver — enforces inlet/outlet/wall/symmetry logic on ghost cells

from typing import List, Set
from src.grid_modules.cell import Cell

def apply_boundary_conditions(grid: List[Cell], ghost_registry: Set[int], config: dict) -> List[Cell]:
    """
    Applies physical boundary conditions to ghost cells based on boundary tags.

    Args:
        grid (List[Cell]): Grid including ghost cells.
        ghost_registry (Set[int]): Identifiers for ghost cells.
        config (dict): Full simulation config block including boundary conditions.

    Returns:
        List[Cell]: Grid with enforced ghost field logic.
    """
    boundary_cfg = config.get("boundary_conditions", {})
    enforced_velocity = boundary_cfg.get("velocity", None)
    enforced_pressure = boundary_cfg.get("pressure", None)
    apply_to = boundary_cfg.get("apply_to", [])
    no_slip = boundary_cfg.get("no_slip", False)

    for cell in grid:
        if id(cell) not in ghost_registry:
            continue

        # ✅ Velocity enforcement
        if "velocity" in apply_to and enforced_velocity is not None:
            cell.velocity = [0.0, 0.0, 0.0] if no_slip else enforced_velocity[:]

        # ✅ Pressure enforcement
        if "pressure" in apply_to and isinstance(enforced_pressure, (int, float)):
            cell.pressure = enforced_pressure

        # 💡 Null out any field not explicitly assigned
        if "velocity" not in apply_to:
            cell.velocity = None
        if "pressure" not in apply_to:
            cell.pressure = None

    return grid



