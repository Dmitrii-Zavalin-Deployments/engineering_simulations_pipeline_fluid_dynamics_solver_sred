# src/physics/pressure_methods/jacobi.py
# 🔁 Jacobi iteration for pressure Poisson solve (∇²p = divergence)

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple
import math

def solve_jacobi_pressure(grid: List[Cell], divergence: List[float], config: dict) -> List[float]:
    """
    Solves pressure Poisson equation using Jacobi iteration.

    Args:
        grid (List[Cell]): Grid of cells with fluid_mask and pressure fields
        divergence (List[float]): Divergence values for fluid cells
        config (dict): Simulation config with domain resolution and solver params

    Returns:
        List[float]: Updated pressure values for fluid cells (same order as input)
    """
    # Domain spacing
    domain = config.get("domain_definition", {})
    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / domain.get("nx", 1)
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / domain.get("ny", 1)
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / domain.get("nz", 1)
    spacing_sq = dx * dx  # assuming uniform spacing for now

    # Solver parameters
    solver_cfg = config.get("pressure_solver", {})
    max_iter = solver_cfg.get("max_iterations", 100)
    tolerance = solver_cfg.get("tolerance", 1e-6)

    # Index fluid cells
    fluid_coords: List[Tuple[float, float, float]] = []
    fluid_map: Dict[Tuple[float, float, float], float] = {}
    divergence_map: Dict[Tuple[float, float, float], float] = {}

    fluid_idx = 0
    for cell in grid:
        if cell.fluid_mask:
            coord = (cell.x, cell.y, cell.z)
            fluid_coords.append(coord)
            fluid_map[coord] = cell.pressure  # preserve initial pressure
            if fluid_idx < len(divergence):
                divergence_map[coord] = divergence[fluid_idx]
                fluid_idx += 1

    if len(fluid_coords) != len(divergence_map):
        raise ValueError("Mismatch between fluid cells and divergence entries")

    def get_pressure(coord: Tuple[float, float, float]) -> float:
        return fluid_map.get(coord, 0.0)

    # Jacobi iterations
    for iteration in range(max_iter):
        new_map = {}
        max_residual = 0.0

        for coord in fluid_coords:
            x, y, z = coord
            sum_neighbors = 0.0
            for offset in [(-dx, 0, 0), (dx, 0, 0), (0, -dy, 0), (0, dy, 0), (0, 0, -dz), (0, 0, dz)]:
                neighbor = (x + offset[0], y + offset[1], z + offset[2])
                if neighbor in fluid_map:
                    sum_neighbors += get_pressure(neighbor)
                else:
                    sum_neighbors += get_pressure(coord)  # Neumann BC approximation

            new_p = (sum_neighbors - spacing_sq * divergence_map[coord]) / 6.0
            residual = abs(new_p - fluid_map[coord])
            max_residual = max(max_residual, residual)
            new_map[coord] = new_p

        fluid_map = new_map

        if max_residual < tolerance:
            break

    return [fluid_map[coord] for coord in fluid_coords]



