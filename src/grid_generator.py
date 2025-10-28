# src/grid_generator.py
# 🧱 Grid Generator — initializes spatial topology, fluid masking, and boundary
# tags
# 📌 This module builds structured grids with optional geometry-based fluid
# masking.
# It excludes only cells explicitly marked fluid_mask=False.
# It does NOT skip based on adjacency or ghost proximity — all logic is
# geometry-mask-driven.

import logging
from src.grid_modules.cell import Cell
from src.grid_modules.grid_geometry import generate_coordinates
from src.grid_modules.initial_field_assigner import assign_fields
from src.grid_modules.boundary_manager import apply_boundaries
from src.utils.mask_interpreter import decode_geometry_mask_flat

# ✅ Centralized debug flag for GitHub Actions logging
debug = True


def generate_grid(domain: dict, initial_conditions: dict) -> list[Cell]:
    """
    [LEGACY] Generates a structured 3D grid with seeded velocity/pressure and
    tagged boundaries.
    This does not apply geometry-based fluid masking.

    Roadmap Alignment:
    - Domain resolution → grid_geometry.py
    - Initial velocity/pressure → initial_field_assigner.py
    - Boundary tagging → boundary_manager.py

    Purpose:
    - Initializes full fluid domain without topology masking
    - Used for simple test cases or full-domain fluid simulations
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
    if not coordinates and debug:
        logging.warning(
            "⚠️ Empty grid generated — no spatial cells due to zero resolution"
        )

    seeded_cells = assign_fields([
        Cell(x, y, z, velocity=[], pressure=0.0, fluid_mask=True)
        for (_, _, _, x, y, z) in coordinates
    ], initial_conditions)

    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells


def generate_grid_with_mask(
    domain: dict,
    initial_conditions: dict,
    geometry: dict
) -> list[Cell]:
    """
    Generates a 3D grid with seeded velocity/pressure, boundary tags, and fluid
    masking.
    Fluid masking is extracted using geometry_mask_flat and geometry_mask_shape.

    Roadmap Alignment:
    - Domain resolution → grid_geometry.py
    - Fluid topology → mask_interpreter.py
    - Initial velocity/pressure → initial_field_assigner.py
    - Boundary tagging → boundary_manager.py

    Purpose:
    - Initializes spatial grid with encoded fluid vs solid topology
    - Supports reflex scoring, ghost logic, and continuity enforcement
    """
    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]
    shape = geometry["geometry_mask_shape"]
    if shape != [nx, ny, nz]:
        raise ValueError(
            f"Geometry mask shape {shape} does not match domain resolution "
            f"[{nx}, {ny}, {nz}]"
        )

    flat_mask = geometry["geometry_mask_flat"]
    encoding = geometry.get("mask_encoding", {"fluid": 1, "solid": 0})
    order = geometry.get("flattening_order", "x-major")

    try:
        fluid_mask_list = decode_geometry_mask_flat(
            flat_mask, [nx, ny, nz], encoding, order
        )
    except Exception as e:
        raise ValueError(f"❌ Failed to decode fluid mask: {e}")

    coordinates = generate_coordinates(domain)
    if not coordinates and debug:
        logging.warning(
            "⚠️ Empty grid generated — no spatial cells due to zero resolution"
        )

    if len(fluid_mask_list) != len(coordinates):
        raise ValueError(
            f"❌ Geometry mask length {len(fluid_mask_list)} does not match "
            f"coordinate count {len(coordinates)}"
        )

    cells = []
    for idx, (ix, iy, iz, x, y, z) in enumerate(coordinates):
        fluid_mask = fluid_mask_list[idx]
        velocity = [] if fluid_mask else None
        pressure = 0.0 if fluid_mask else None
        cells.append(Cell(
            x, y, z,
            velocity=velocity,
            pressure=pressure,
            fluid_mask=fluid_mask
        ))

    seeded_cells = assign_fields(cells, initial_conditions)
    tagged_cells = apply_boundaries(seeded_cells, domain)

    return tagged_cells
