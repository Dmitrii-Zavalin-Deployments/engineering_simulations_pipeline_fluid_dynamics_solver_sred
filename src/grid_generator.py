# src/grid_generator.py

import logging
from src.grid_modules.cell import Cell
from src.grid_modules.grid_geometry import generate_coordinates
from src.grid_modules.initial_field_assigner import assign_fields
from src.grid_modules.boundary_manager import apply_boundaries
from src.utils.mask_interpreter import decode_geometry_mask_flat

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
        Cell(x, y, z, velocity=[], pressure=0.0, fluid_mask=True)
        for (_, _, _, x, y, z) in coordinates
    ], initial_conditions)

    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells


def generate_grid_with_mask(domain: dict, initial_conditions: dict, geometry: dict) -> list[Cell]:
    """
    Generates a 3D grid with seeded velocity/pressure, boundary tags, and fluid masking.
    Fluid masking is extracted using geometry_mask_flat and geometry_mask_shape.

    Args:
        domain (dict): Domain bounds and resolution
        initial_conditions (dict): Initial velocity and pressure
        geometry (dict): Geometry mask configuration block from input JSON

    Returns:
        list[Cell]: Grid tagged with fluid_mask and boundary conditions
    """
    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]
    shape = geometry["geometry_mask_shape"]
    if shape != [nx, ny, nz]:
        raise ValueError(f"Geometry mask shape {shape} does not match domain resolution [{nx}, {ny}, {nz}]")

    flat_mask = geometry["geometry_mask_flat"]
    encoding = geometry.get("mask_encoding", {"fluid": 1, "solid": 0})
    order = geometry.get("flattening_order", "x-major")

    try:
        fluid_mask_list = decode_geometry_mask_flat(flat_mask, [nx, ny, nz], encoding, order)
    except Exception as e:
        raise ValueError(f"❌ Failed to decode fluid mask: {e}")

    coordinates = generate_coordinates(domain)
    if not coordinates:
        logging.warning("⚠️ Empty grid generated — no spatial cells due to zero resolution")

    if len(fluid_mask_list) != len(coordinates):
        raise ValueError(f"❌ Geometry mask length {len(fluid_mask_list)} does not match coordinate count {len(coordinates)}")

    cells = []
    for idx, (ix, iy, iz, x, y, z) in enumerate(coordinates):
        fluid_mask = fluid_mask_list[idx]
        velocity = [] if fluid_mask else None
        pressure = 0.0 if fluid_mask else None
        cells.append(Cell(x, y, z, velocity=velocity, pressure=pressure, fluid_mask=fluid_mask))

    seeded_cells = assign_fields(cells, initial_conditions)
    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells



