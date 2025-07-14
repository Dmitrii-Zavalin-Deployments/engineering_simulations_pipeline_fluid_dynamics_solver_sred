# src/physics/boundary_condition_solver.py
# 🧪 Boundary Condition Solver — enforces inlet/outlet/wall/symmetry logic on ghost cells and adjacent fluid cells

from typing import List, Set, Dict, Tuple
from src.grid_modules.cell import Cell

def apply_boundary_conditions(grid: List[Cell], ghost_registry: Dict[int, dict], config: dict) -> List[Cell]:
    """
    Applies physical boundary conditions to ghost cells and adjacent fluid cells based on boundary tags.

    Args:
        grid (List[Cell]): Grid including ghost cells.
        ghost_registry (Dict[int, dict]): Metadata for ghost cells with face, origin, pressure, velocity.
        config (dict): Full simulation config block including boundary conditions.

    Returns:
        List[Cell]: Grid with enforced ghost field logic and updated adjacent fluid cells.
    """
    boundary_cfg = config.get("boundary_conditions", {})
    enforced_velocity = boundary_cfg.get("velocity", None)
    enforced_pressure = boundary_cfg.get("pressure", None)
    apply_to = boundary_cfg.get("apply_to", [])
    no_slip = boundary_cfg.get("no_slip", False)

    ghost_ids = set(ghost_registry.keys())

    # ✅ Apply boundary fields to ghost cells
    for cell in grid:
        if id(cell) not in ghost_ids:
            continue

        ghost_applied = False

        # Velocity enforcement
        if "velocity" in apply_to and enforced_velocity is not None:
            cell.velocity = [0.0, 0.0, 0.0] if no_slip else enforced_velocity[:]
            ghost_applied = True
        elif "velocity" not in apply_to:
            cell.velocity = None

        # Pressure enforcement
        if "pressure" in apply_to and isinstance(enforced_pressure, (int, float)):
            cell.pressure = enforced_pressure
            ghost_applied = True
        elif "pressure" not in apply_to:
            cell.pressure = None

        # ✅ Tag ghost as enforced for diagnostics
        if ghost_applied:
            cell.ghost_field_applied = True

    # ✅ Reflect enforcement into adjacent fluid cells using ghost origin mapping
    origin_map = {
        meta["origin"]: meta for meta in ghost_registry.values()
        if isinstance(meta.get("origin"), tuple)
    }

    fluid_map = {
        (cell.x, cell.y, cell.z): cell for cell in grid
        if getattr(cell, "fluid_mask", False)
    }

    for origin_coord, meta in origin_map.items():
        fluid_cell = fluid_map.get(origin_coord)
        if not fluid_cell:
            continue

        # Apply velocity from config
        if "velocity" in apply_to and enforced_velocity is not None:
            if no_slip:
                fluid_cell.velocity = [0.0, 0.0, 0.0]
            elif isinstance(enforced_velocity, list):
                fluid_cell.velocity = enforced_velocity[:]

        # Apply pressure from config
        if "pressure" in apply_to and isinstance(enforced_pressure, (int, float)):
            fluid_cell.pressure = enforced_pressure

    return grid



