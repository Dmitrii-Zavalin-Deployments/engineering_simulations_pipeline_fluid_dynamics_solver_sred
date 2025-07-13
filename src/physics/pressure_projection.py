# src/physics/pressure_projection.py
# 🔁 Pressure projection module for enforcing incompressibility

from src.grid_modules.cell import Cell
from typing import List
from src.physics.pressure_methods.jacobi import solve_jacobi_pressure
from src.physics.pressure_methods.utils import index_fluid_cells, flatten_pressure_field

def solve_pressure_poisson(grid: List[Cell], divergence: List[float], config: dict) -> List[Cell]:
    """
    Computes updated pressure values for fluid cells using the selected solver method.

    Args:
        grid (List[Cell]): Grid of cells with velocity and pressure fields
        divergence (List[float]): Divergence values at each fluid cell
        config (dict): Simulation config including solver parameters

    Returns:
        List[Cell]: Grid with updated pressure values (fluid cells only)
    """
    method = config.get("pressure_solver", {}).get("method", "jacobi").lower()

    # 🔍 Index fluid cells for solver
    fluid_coords = index_fluid_cells(grid)
    fluid_cell_count = len(fluid_coords)

    if len(divergence) != fluid_cell_count:
        raise ValueError(
            f"Divergence list length ({len(divergence)}) does not match number of fluid cells ({fluid_cell_count})"
        )

    # 🔁 Select solver method
    if method == "jacobi":
        pressure_values = solve_jacobi_pressure(grid, divergence, config)
    else:
        raise ValueError(f"Unknown or unsupported pressure solver method: '{method}'")

    # 🧱 Reconstruct grid with updated pressures
    updated = []
    fluid_index = 0
    for cell in grid:
        if cell.fluid_mask:
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=cell.velocity[:] if isinstance(cell.velocity, list) else None,
                pressure=pressure_values[fluid_index],
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

    return updated



