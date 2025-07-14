# src/physics/pressure_projection.py
# 🔁 Pressure projection module for enforcing incompressibility

from typing import List, Tuple, Set
from src.grid_modules.cell import Cell
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure
from src.physics.pressure_methods.utils import index_fluid_cells, flatten_pressure_field
from src.physics.velocity_projection import apply_pressure_velocity_projection

def extract_ghost_coords(grid: List[Cell]) -> Set[Tuple[float, float, float]]:
    """
    Extract coordinates of ghost cells in the grid.

    Args:
        grid (List[Cell]): Full grid including fluid and ghost cells

    Returns:
        Set[Tuple]: Coordinates of ghost cells
    """
    return {
        (cell.x, cell.y, cell.z)
        for cell in grid
        if not cell.fluid_mask and hasattr(cell, "ghost_face")
    }

def solve_pressure_poisson(grid: List[Cell], divergence: List[float], config: dict) -> Tuple[List[Cell], bool]:
    """
    Computes updated pressure values for fluid cells using the selected solver method,
    then projects velocity to enforce incompressibility.

    Args:
        grid (List[Cell]): Grid of cells with velocity and pressure fields
        divergence (List[float]): Divergence values at each fluid cell
        config (dict): Simulation config including solver parameters

    Returns:
        Tuple[List[Cell], bool]: 
            - Grid with updated pressure and velocity values
            - pressure_mutated flag indicating if any fluid pressure changed
    """
    method = config.get("pressure_solver", {}).get("method", "jacobi").lower()

    # 🔍 Index fluid cells for solver
    fluid_coords = index_fluid_cells(grid)
    fluid_cell_count = len(fluid_coords)

    if len(divergence) != fluid_cell_count:
        raise ValueError(
            f"Divergence list length ({len(divergence)}) does not match number of fluid cells ({fluid_cell_count})"
        )

    # 🧱 Prepare ghost info
    ghost_coords = extract_ghost_coords(grid)

    # 🔁 Select solver method
    if method == "jacobi":
        pressure_values = solve_jacobi_pressure(grid, divergence, config, ghost_coords)
    else:
        raise ValueError(f"Unknown or unsupported pressure solver method: '{method}'")

    # 🧱 Reconstruct grid and track pressure mutation
    updated = []
    fluid_index = 0
    pressure_mutated = False
    for cell in grid:
        if cell.fluid_mask:
            new_pressure = pressure_values[fluid_index]
            if isinstance(cell.pressure, float) and abs(cell.pressure - new_pressure) > 1e-6:
                pressure_mutated = True
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=cell.velocity[:] if isinstance(cell.velocity, list) else None,
                pressure=new_pressure,
                fluid_mask=True
            )
            fluid_index += 1
        else:
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=None,
                pressure=None,
                fluid_mask=False
            )
        updated.append(updated_cell)

    # 💨 Apply pressure-based velocity projection
    projected_grid = apply_pressure_velocity_projection(updated, config)

    return projected_grid, pressure_mutated



