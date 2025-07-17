# src/physics/ghost_influence_applier.py
# 🧱 Ghost Influence Applier — applies pressure/velocity from ghosts to adjacent fluid cells

import logging
from typing import List, Tuple
from src.grid_modules.cell import Cell

logger = logging.getLogger(__name__)

def fuzzy_equal(v1: List[float], v2: List[float], tol: float = 1e-6) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(v1, v2))

def apply_ghost_influence(
    grid: List[Cell],
    spacing: Tuple[float, float, float],
    verbose: bool = False,
    radius: int = 1
) -> int:
    dx, dy, dz = spacing
    tol = 1e-6
    influence_count = 0
    bordering_fluid_count = 0
    skipped_due_to_match = 0

    fluid_cells = [c for c in grid if getattr(c, "fluid_mask", False)]
    fluid_coord_map = {
        (round(c.x, 6), round(c.y, 6), round(c.z, 6)): c
        for c in fluid_cells
    }

    ghost_cells = [c for c in grid if not getattr(c, "fluid_mask", True)]

    def coords_are_neighbors(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        return (
            abs(a[0] - b[0]) <= dx * radius + tol and
            abs(a[1] - b[1]) <= dy * radius + tol and
            abs(a[2] - b[2]) <= dz * radius + tol
        )

    for ghost in ghost_cells:
        ghost_coord = (round(ghost.x, 6), round(ghost.y, 6), round(ghost.z, 6))

        if verbose:
            print(f"[ghost_gen] ghost.cell @ {ghost_coord} → velocity={ghost.velocity}, pressure={ghost.pressure}")

        for f_coord, fluid_cell in fluid_coord_map.items():
            if coords_are_neighbors(ghost_coord, f_coord):
                bordering_fluid_count += 1
                fluid_cell.ghost_influence_attempted = True
                modified = False

                velocity_match = isinstance(ghost.velocity, list) and fuzzy_equal(ghost.velocity, fluid_cell.velocity)
                pressure_match = isinstance(ghost.pressure, (int, float)) and abs(ghost.pressure - fluid_cell.pressure) < tol

                if isinstance(ghost.velocity, list) and not velocity_match:
                    fluid_cell.velocity = ghost.velocity[:]
                    modified = True

                if isinstance(ghost.pressure, (int, float)) and not pressure_match:
                    fluid_cell.pressure = ghost.pressure
                    modified = True

                if modified:
                    fluid_cell.influenced_by_ghost = True
                    fluid_cell.ghost_influence_applied = True
                    influence_count += 1
                    if verbose:
                        print(f"[DEBUG] Ghost @ {ghost_coord} → influenced fluid @ {f_coord}")
                else:
                    fluid_cell.influence_skipped_due_to_match = True
                    fluid_cell.triggered_by = "ghost adjacency — no mutation (fields matched)"
                    if verbose and velocity_match and pressure_match:
                        skipped_due_to_match += 1
                        print(f"Influence skipped: matched fields → ghost={ghost.velocity}, fluid={fluid_cell.velocity}")
                    logger.debug(f"[influence] ghost.v={ghost.velocity}, fluid.v={fluid_cell.velocity}")
                    logger.debug(f"[influence] suppression reason: fields matched within tolerance")

                # ✅ Optional audit: log adjacency context even if no pressure mutation occurred
                logger.debug(f"[audit] Ghost→Fluid neighbor match @ {f_coord}, enforced={modified}")

    if verbose:
        print(f"[DEBUG] Total fluid cells influenced by ghosts: {influence_count}")
        print(f"[DEBUG] Total fluid cells adjacent to ghosts: {bordering_fluid_count}")
        if bordering_fluid_count > 0 and influence_count == 0:
            print("⚠️ Ghosts adjacent to fluid cells did not trigger influence propagation.")
        if skipped_due_to_match > 0:
            print(f"[DEBUG] Skipped influence due to field match: {skipped_due_to_match} fluid cells")

    return influence_count



