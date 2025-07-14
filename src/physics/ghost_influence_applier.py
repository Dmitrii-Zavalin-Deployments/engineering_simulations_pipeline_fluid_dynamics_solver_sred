# src/physics/ghost_influence_applier.py
# 🧱 Ghost Influence Applier — applies pressure/velocity from ghosts to adjacent fluid cells

from typing import List, Tuple
from src.grid_modules.cell import Cell

def apply_ghost_influence(grid: List[Cell], spacing: Tuple[float, float, float], verbose: bool = False) -> int:
    """
    Applies ghost cell pressure and velocity to adjacent fluid cells.
    For now: transfers velocity and pressure directly if ghost has value and fluid does not.

    Args:
        grid (List[Cell]): Full simulation grid including fluid and ghost cells
        spacing (tuple): (dx, dy, dz) physical spacing
        verbose (bool): If True, print influence mapping per ghost

    Returns:
        int: Number of fluid cells modified by ghost influence
    """
    dx, dy, dz = spacing
    tol = 1e-6
    influence_count = 0

    # Build coordinate index for fluid cells
    fluid_cells = [c for c in grid if getattr(c, "fluid_mask", False)]
    fluid_coord_map = {
        (round(c.x, 6), round(c.y, 6), round(c.z, 6)): c
        for c in fluid_cells
    }

    # Build list of ghosts
    ghost_cells = [c for c in grid if not getattr(c, "fluid_mask", True)]

    # Helper: check if two coords are neighbors
    def coords_are_neighbors(a, b):
        return (
            abs(a[0] - b[0]) <= dx + tol and
            abs(a[1] - b[1]) <= dy + tol and
            abs(a[2] - b[2]) <= dz + tol
        )

    for ghost in ghost_cells:
        gx, gy, gz = ghost.x, ghost.y, ghost.z
        ghost_coord = (round(gx, 6), round(gy, 6), round(gz, 6))

        for f_coord, fluid_cell in fluid_coord_map.items():
            if coords_are_neighbors(ghost_coord, f_coord):
                modified = False

                # Apply velocity if fluid cell has None or zero velocity
                if fluid_cell.velocity == [0.0, 0.0, 0.0] and ghost.velocity != [0.0, 0.0, 0.0]:
                    fluid_cell.velocity = ghost.velocity[:]
                    modified = True

                # Apply pressure if fluid pressure is None and ghost has pressure
                if fluid_cell.pressure in [None, 0.0] and isinstance(ghost.pressure, (int, float)):
                    fluid_cell.pressure = ghost.pressure
                    modified = True

                if modified:
                    influence_count += 1
                    if verbose:
                        print(f"[DEBUG] Ghost @ ({gx:.2f}, {gy:.2f}, {gz:.2f}) → modified fluid @ {f_coord}")

    if verbose:
        print(f"[DEBUG] Total fluid cells influenced by ghosts: {influence_count}")
    return influence_count



