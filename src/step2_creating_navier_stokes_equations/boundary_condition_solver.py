# src/physics/boundary_condition_solver.py
# 🧪 Boundary Condition Solver — enforces inlet/outlet/wall/symmetry logic on ghost cells and adjacent fluid cells
# 📌 This module enforces Dirichlet and Neumann conditions on ghost cells and reflects changes into adjacent fluid cells.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or boundary proximity — all logic is geometry-mask-driven.

from typing import List, Dict
from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

def apply_boundary_conditions(grid: List[Cell], ghost_registry: Dict[int, dict], config: dict) -> List[Cell]:
    """
    Applies physical boundary conditions to ghost cells and adjacent fluid cells.

    Roadmap Alignment:
    - Governing Equations:
        - Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u
        - Continuity: ∇ · u = 0

    Boundary enforcement modifies ghost cells to reflect:
        - Dirichlet conditions: fixed velocity or pressure
        - Neumann conditions: zero-gradient (handled in pressure solver)
        - No-slip walls: velocity = 0 at solid boundaries

    These conditions ensure correct momentum transfer and pressure coupling at domain boundaries.

    Args:
        grid (List[Cell]): Grid including ghost cells.
        ghost_registry (Dict[int, dict]): Metadata for ghost cells with face, origin, pressure, velocity.
        config (dict): Full simulation config block including boundary conditions.

    Returns:
        List[Cell]: Grid with enforced ghost field logic and updated adjacent fluid cells.
    """
    boundary_blocks = config.get("boundary_conditions", [])
    if not isinstance(boundary_blocks, list):
        if debug:
            print(f"[BOUNDARY] ⚠️ Unexpected boundary_conditions format: {type(boundary_blocks)} → expected list of dicts")
        return grid

    ghost_ids = set(ghost_registry.keys())

    for bc in boundary_blocks:
        if not isinstance(bc, dict):
            if debug:
                print(f"[BOUNDARY] ⚠️ Skipping malformed boundary block: {type(bc)} → {bc}")
            continue

        enforced_velocity = bc.get("velocity", None)
        enforced_pressure = bc.get("pressure", None)
        apply_to = bc.get("apply_to", [])
        no_slip = bc.get("no_slip", False)

        # ✅ Step 1: Apply boundary fields to ghost cells
        for cell in grid:
            if id(cell) not in ghost_ids:
                continue

            if "velocity" in apply_to and enforced_velocity is not None:
                cell.velocity = [0.0, 0.0, 0.0] if no_slip else enforced_velocity[:]
            elif "velocity" not in apply_to:
                cell.velocity = None

            if "pressure" in apply_to and isinstance(enforced_pressure, (int, float)):
                cell.pressure = enforced_pressure
            elif "pressure" not in apply_to:
                cell.pressure = None

            if debug:
                print(f"[BOUNDARY] Ghost cell updated → velocity={cell.velocity}, pressure={cell.pressure}")

        # ✅ Step 2: Reflect enforcement into adjacent fluid cells using ghost origin mapping
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

            if "velocity" in apply_to and enforced_velocity is not None:
                if no_slip:
                    fluid_cell.velocity = [0.0, 0.0, 0.0]
                elif isinstance(enforced_velocity, list):
                    fluid_cell.velocity = enforced_velocity[:]

            if "pressure" in apply_to and isinstance(enforced_pressure, (int, float)):
                fluid_cell.pressure = enforced_pressure

            if debug:
                print(f"[BOUNDARY] Fluid cell @ {origin_coord} updated → velocity={fluid_cell.velocity}, pressure={fluid_cell.pressure}")

    return grid



