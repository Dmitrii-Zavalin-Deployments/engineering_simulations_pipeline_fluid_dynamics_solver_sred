# src/grid_modules/initial_field_assigner.py

from src.grid_modules.cell import Cell

def assign_fields(cells: list[Cell], initial_conditions: dict) -> list[Cell]:
    """
    Assigns initial velocity and pressure to fluid cells.
    If fluid_mask is missing, cells default to fluid=True.
    Solid and ghost cells (fluid_mask=False) are sanitized with None values.
    Raises ValueError if required fields are missing or invalid.

    Args:
        cells (list[Cell]): Grid cells to initialize
        initial_conditions (dict): Must contain "initial_velocity" and "initial_pressure"

    Returns:
        list[Cell]: Cells with velocity and pressure assigned or suppressed
    """
    if "initial_velocity" not in initial_conditions:
        raise ValueError("❌ Missing 'initial_velocity' in initial_conditions")

    if "initial_pressure" not in initial_conditions:
        raise ValueError("❌ Missing 'initial_pressure' in initial_conditions")

    velocity = initial_conditions["initial_velocity"]
    pressure = initial_conditions["initial_pressure"]

    if not isinstance(velocity, list) or len(velocity) != 3 or not all(isinstance(v, (int, float)) for v in velocity):
        raise ValueError("❌ 'initial_velocity' must be a list of 3 numeric components")

    if not isinstance(pressure, (int, float)):
        raise ValueError("❌ 'initial_pressure' must be a numeric value")

    for cell in cells:
        if not hasattr(cell, "fluid_mask") or cell.fluid_mask:
            cell.velocity = velocity[:]
            cell.pressure = pressure
        else:
            cell.velocity = None
            cell.pressure = None

    return cells



