# src/grid_generator.py

import logging
import numpy as np
from src.grid_modules.cell import Cell
from src.grid_modules.grid_geometry import generate_coordinates
from src.grid_modules.initial_field_assigner import assign_fields
from src.grid_modules.boundary_manager import apply_boundaries

def generate_grid(domain: dict, initial_conditions: dict) -> list[Cell]:
    """
    [LEGACY] Generates a structured 3D grid with seeded velocity/pressure and tagged boundaries.
    This does not apply geometry-based fluid masking.

    Args:
        domain (dict): Includes physical bounds and resolution
        initial_conditions (dict): Input velocity and pressure

    Returns:
        list[Cell]: Final grid
    """
    required_keys = [
        "min_x", "max_x", "nx",
        "min_y", "max_y", "ny",
        "min_z", "max_z", "nz"
    ]
    missing = [key for key in required_keys if key not in domain]
    if missing:
        raise ValueError(f"Missing domain keys: {missing}")

    coordinates = generate_coordinates(domain)
    if not coordinates:
        logging.warning("⚠️ Empty grid generated — no spatial cells due to zero resolution")

    seeded_cells = assign_fields([
        Cell(x, y, z, velocity=[], pressure=0.0) for (x, y, z) in coordinates
    ], initial_conditions)

    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells


def generate_grid_with_mask(domain: dict, initial_conditions: dict, geometry: dict) -> list[Cell]:
    """
    Generates a 3D grid with seeded velocity/pressure, boundary tags, and fluid masking.
    Fluid masking is extracted from geometry_mask_flat and geometry_mask_shape.

    Args:
        domain (dict): Domain bounds and resolution
        initial_conditions (dict): Initial velocity and pressure
        geometry (dict): Geometry mask configuration block from input JSON

    Returns:
        list[Cell]: Grid tagged with fluid_mask and boundary conditions
    """
    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]
    flat_mask = geometry["geometry_mask_flat"]
    shape = geometry["geometry_mask_shape"]
    fluid_value = geometry["mask_encoding"]["fluid"]

    # 🧩 Ensure shape matches expected domain resolution
    if shape != [nx, ny, nz]:
        raise ValueError(f"Geometry mask shape {shape} does not match domain resolution [{nx}, {ny}, {nz}]")

    # 🧬 Reshape flat mask into a 3D cube: [z][y][x]
    try:
        mask_array = np.array(flat_mask).reshape((nz, ny, nx))
    except Exception as e:
        raise ValueError(f"Failed to reshape geometry_mask_flat: {e}")

    coordinates = generate_coordinates(domain)
    if not coordinates:
        logging.warning("⚠️ Empty grid generated — no spatial cells")

    cells = []
    for (x, y, z) in coordinates:
        fluid_mask = mask_array[z][y][x] == fluid_value
        cells.append(Cell(x, y, z, velocity=[], pressure=0.0, fluid_mask=fluid_mask))

    # 🧬 Assign velocity/pressure fields
    seeded_cells = assign_fields(cells, initial_conditions)

    # 🧱 Apply boundary tagging
    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells



