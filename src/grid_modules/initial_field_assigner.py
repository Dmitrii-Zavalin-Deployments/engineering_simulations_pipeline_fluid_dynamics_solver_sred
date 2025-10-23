# src/grid_modules/initial_field_assigner.py
# 🌀 Initial Field Assigner — seeds velocity and pressure for ∂u/∂t initialization and reflex diagnostics
# 📌 This module assigns initial conditions to fluid cells only.
# It enforces that only fluid_mask=True cells receive velocity and pressure.
# Solid and ghost cells are sanitized to None.

from src.grid_modules.cell import Cell

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def assign_fields(cells: list[Cell], initial_conditions: dict) -> list[Cell]:
    """
    Assigns initial velocity and pressure to fluid cells.

    Roadmap Alignment:
    Governing Equation:
        Momentum: ρ(∂u/∂t + u · ∇u) = -∇P + μ∇²u

    Purpose:
    - Seeds ∂u/∂t initialization for momentum solver
    - Supports reflex diagnostics and mutation tracking
    - Sanitizes solid and ghost cells for boundary enforcement

    Strategy:
    - Fluid cells → velocity[:] and pressure
    - Solid/ghost cells → velocity=None, pressure=None

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
        fluid = getattr(cell, "fluid_mask", True)
        if fluid:
            cell.velocity = velocity[:]
            cell.pressure = pressure
            if debug:
                print(f"[INIT] Fluid @ ({cell.x:.2f}, {cell.y:.2f}, {cell.z:.2f}) → velocity: {velocity}, pressure: {pressure}")
        else:
            cell.velocity = None
            cell.pressure = None

    return cells



