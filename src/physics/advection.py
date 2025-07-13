# src/physics/advection.py
# 🌀 Velocity advection module — uses default Euler method with ghost exclusion and symmetry-ready hooks

from src.grid_modules.cell import Cell
from typing import List, Set, Dict
from src.physics.advection_methods.euler import compute_euler_velocity

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
                # TODO: Implement mirroring logic if needed
                continue  # skip for now
            continue  # skip all ghost cells from advection
        physical_cells.append(cell)

    updated = compute_euler_velocity(physical_cells, dt, config)

    return updated



