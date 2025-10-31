# src/grid_modules/initial_field_assigner.py
# 🌀 Initial Field Assigner — sets velocity and pressure for fluid cells based on config initial_conditions

from typing import List, Dict
from src.grid_modules.cell import Cell
from src.config.config_validator import validate_config

debug = False  # Centralized debug flag

def assign_initial_fields(grid: List[Cell], config: Dict) -> List[Cell]:
    """
    Assigns initial velocity and pressure to all fluid cells in the grid.

    Args:
        grid (List[Cell]): Grid of Cell objects (fluid and ghost)
        config (Dict): Validated simulation configuration

    Returns:
        List[Cell]: Grid with updated velocity and pressure for fluid cells
    """
    validate_config(config)

    init = config["initial_conditions"]
    initial_velocity = init["initial_velocity"]
    initial_pressure = init["initial_pressure"]

    for idx, cell in enumerate(grid):
        if cell.fluid_mask:
            cell.velocity = initial_velocity
            cell.pressure = initial_pressure

            if debug:
                x, y, z = cell.x, cell.y, cell.z
                print(f"[INIT] Fluid cell @ ({x:.2f}, {y:.2f}, {z:.2f}) → velocity={initial_velocity}, pressure={initial_pressure}")

    return grid



