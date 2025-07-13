# src/physics/pressure_methods/jacobi.py
# 🔁 Jacobi iteration for pressure Poisson solve (∇²p = divergence)

from src.grid_modules.cell import Cell
from typing import List, Dict, Tuple
from src.physics.pressure_methods.utils import index_fluid_cells, build_pressure_map
from src.physics.pressure_methods.boundary import handle_solid_neighbors
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
    # 🧮 Grid spacing
    domain = config.get("domain_definition", {})
    dx = (domain.get("max_x", 1.0) - domain.get("min_x", 0.0)) / domain.get("nx", 1)
    dy = (domain.get("max_y", 1.0) - domain.get("min_y", 0.0)) / domain.get("ny", 1)
    dz = (domain.get("max_z", 1.0) - domain.get("min_z", 0.0)) / domain.get("nz", 1)
    spacing_sq = dx * dx  # Assuming uniform spacing

    # 🔧 Solver settings
    solver_cfg = config.get("pressure_solver", {})
    max_iter = solver_cfg.get("max_iterations", 100)
    tolerance = solver_cfg.get("tolerance", 1e-6)

    # 🗂️ Fluid indexing and pressure map
    fluid_coords = index_fluid_cells(grid)
    pressure_map = build_pressure_map(grid)

    # 💧 Divergence mapping
    if len(divergence) != len(fluid_coords):
        raise ValueError("Mismatch between fluid cells and divergence values")
    divergence_map = dict(zip(fluid_coords, divergence))

    # 🗺️ Fluid mask map for neighbor handling
    fluid_mask_map = {
        (cell.x, cell.y, cell.z): cell.fluid_mask for cell in grid
    }

    # 🔁 Jacobi iterations
    for _ in range(max_iter):
        new_map = {}
        max_residual = 0.0

        for coord in fluid_coords:
            x, y, z = coord
            neighbors = [
                (x - dx, y, z), (x + dx, y, z),
                (x, y - dy, z), (x, y + dy, z),
                (x, y, z - dz), (x, y, z + dz)
            ]

            neighbor_sum = handle_solid_neighbors(coord, neighbors, pressure_map, fluid_mask_map)
            rhs = spacing_sq * divergence_map.get(coord, 0.0)
            new_p = (neighbor_sum - rhs) / 6.0

            residual = abs(new_p - pressure_map.get(coord, 0.0))
            max_residual = max(max_residual, residual)
            new_map[coord] = new_p

        pressure_map = new_map

        if max_residual < tolerance:
            break

    # 📦 Flatten output in fluid cell order
    return [pressure_map.get(coord, 0.0) for coord in fluid_coords]



