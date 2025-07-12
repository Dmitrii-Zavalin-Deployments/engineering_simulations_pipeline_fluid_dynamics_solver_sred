# src/physics/viscosity.py
# 💧 Stub: Viscous diffusion module for velocity damping

from src.grid_modules.cell import Cell
from typing import List

def apply_viscous_terms(grid: List[Cell], dt: float, config: dict) -> List[Cell]:
    """
    Placeholder for viscous diffusion logic.

    Args:
        grid (List[Cell]): Current grid with velocity and pressure
        dt (float): Time step duration
        config (dict): Fluid properties and simulation config

    Returns:
        List[Cell]: Grid with viscosity-adjusted velocities (fluid cells only)

    Notes:
        This stub applies no real viscous effects. It simply returns a copy.
        Future implementation will apply Laplacian smoothing for velocity damping
        using neighbor averaging and viscosity coefficient.
    """
    viscosity = config.get("fluid_properties", {}).get("viscosity", 0.0)

    updated = []
    for cell in grid:
        if cell.fluid_mask:
            updated_cell = Cell(
                x=cell.x,
                y=cell.y,
                z=cell.z,
                velocity=cell.velocity[:],  # unchanged for now
                pressure=cell.pressure,
                fluid_mask=True
            )
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



