# src/physics/advection.py
# 🌀 Velocity advection module — uses default Euler method with ghost exclusion and symmetry-ready hooks

from src.grid_modules.cell import Cell
from typing import List, Set, Dict
from src.physics.advection_methods.euler import compute_euler_velocity
import math

def compute_advection(grid: List[Cell],
                      dt: float,
                      config: dict,
                      ghost_registry: Set[int] = set(),
                      ghost_metadata: Dict[int, Dict] = {}) -> List[Cell]:
    """
    Evolves fluid velocity fields using the Forward Euler method,
    excluding ghost cells and optionally applying symmetry enforcement.

    Args:
        grid (List[Cell]): List of Cell objects with current velocity
        dt (float): Time step (delta t)
        config (dict): Simulation config
        ghost_registry (Set[int]): IDs of ghost cells to exclude from advection
        ghost_metadata (Dict[int, Dict]): Optional metadata for mirroring or boundary-aware logic

    Returns:
        List[Cell]: Grid with updated velocity values (ghosts skipped)
    """
    physical_cells = []

    for cell in grid:
        if id(cell) in ghost_registry:
            meta = ghost_metadata.get(id(cell), {})
            if meta.get("boundary_type") == "symmetry":
                # Future: Implement symmetry mirror logic
                continue
            continue
        physical_cells.append(cell)

    # 🔁 Compute velocity updates via Euler method
    updated_cells = compute_euler_velocity(physical_cells, dt, config)

    # 🧪 Diagnostic: Track mutation per fluid cell
    mutation_count = 0
    for before, after in zip(physical_cells, updated_cells):
        if before.fluid_mask:
            v0 = before.velocity if isinstance(before.velocity, list) else [0.0, 0.0, 0.0]
            v1 = after.velocity if isinstance(after.velocity, list) else [0.0, 0.0, 0.0]
            delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v0)))
            if delta > 1e-8:
                mutation_count += 1

    print(f"📦 Advection applied to {len(updated_cells)} physical cells → {mutation_count} velocity mutations")

    return updated_cells



