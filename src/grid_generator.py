# src/grid_generator.py

import logging
from src.grid_modules.cell import Cell
from src.grid_modules.grid_geometry import generate_coordinates
from src.grid_modules.initial_field_assigner import assign_fields
from src.grid_modules.boundary_manager import apply_boundaries

def generate_grid(domain: dict, initial_conditions: dict) -> list[Cell]:
    """
    Generates a structured 3D grid with seeded velocity/pressure and tagged boundaries.
    Performs domain validation before grid construction.

    Args:
        domain (dict): Must include physical bounds and resolution
        initial_conditions (dict): Input velocity and pressure

    Returns:
        list[Cell]: Final grid with fields and boundary metadata

    Raises:
        ValueError: If domain is missing required keys or contains invalid values
    """
    # 🔎 Validate domain schema
    required_keys = [
        "min_x", "max_x", "nx",
        "min_y", "max_y", "ny",
        "min_z", "max_z", "nz"
    ]
    missing = [key for key in required_keys if key not in domain]
    if missing:
        raise ValueError(f"Missing domain keys: {missing}")

    try:
        coordinates = generate_coordinates(domain)
    except Exception as e:
        logging.error(f"Domain coordinate generation failed: {e}")
        raise

    if not coordinates:
        logging.warning("⚠️ Empty grid generated — no spatial cells due to zero resolution")

    # 🧬 Apply validated field values
    try:
        seeded_cells = assign_fields([
            Cell(x, y, z, velocity=[], pressure=0.0) for (x, y, z) in coordinates
        ], initial_conditions)
    except ValueError as e:
        logging.warning(f"Initial field assignment failed: {e}")
        seeded_cells = [
            Cell(x, y, z, velocity=[0.0, 0.0, 0.0], pressure=0.0) for (x, y, z) in coordinates
        ]

    # 🧱 Apply boundary tagging
    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells



