# src/grid_modules/grid_geometry.py
# 🌐 Grid Geometry Generator — computes grid indices and physical coordinates for simulation domain
# 📌 This module initializes the spatial layout of the simulation grid.
# It does NOT interact with fluid_mask or geometry masking logic.
# It is NOT responsible for solver inclusion/exclusion decisions.

import logging

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

def generate_coordinates(domain: dict) -> list[tuple[int, int, int, float, float, float]]:
    """
    Generates 3D grid positions along with physical coordinates.

    Each entry is a tuple: (ix, iy, iz, x, y, z), where:
    - (ix, iy, iz) are grid indices
    - (x, y, z) are physical coordinates based on domain bounds

    Args:
        domain (dict): Must include:
            "min_x", "max_x", "nx",
            "min_y", "max_y", "ny",
            "min_z", "max_z", "nz"

    Returns:
        list of tuple: Grid indices and physical coordinates per cell

    Raises:
        ValueError: If domain parameters are missing or invalid
    """
    required_keys = [
        "min_x", "max_x", "nx",
        "min_y", "max_y", "ny",
        "min_z", "max_z", "nz"
    ]
    missing = [key for key in required_keys if key not in domain]
    if missing:
        raise ValueError(f"Missing domain keys: {missing}")

    try:
        min_x, max_x, nx = float(domain["min_x"]), float(domain["max_x"]), int(domain["nx"])
        min_y, max_y, ny = float(domain["min_y"]), float(domain["max_y"]), int(domain["ny"])
        min_z, max_z, nz = float(domain["min_z"]), float(domain["max_z"]), int(domain["nz"])
    except (TypeError, ValueError) as e:
        logging.error(f"Invalid domain value: {e}")
        raise ValueError("Domain must contain numeric bounds and integer resolution values")

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Resolution values must be greater than zero")

    dx = (max_x - min_x) / nx
    dy = (max_y - min_y) / ny
    dz = (max_z - min_z) / nz

    coordinates = []
    for ix in range(nx):
        x = min_x + (ix + 0.5) * dx
        for iy in range(ny):
            y = min_y + (iy + 0.5) * dy
            for iz in range(nz):
                z = min_z + (iz + 0.5) * dz
                coordinates.append((ix, iy, iz, x, y, z))

    if debug:
        print(f"[GEOMETRY] Generated {len(coordinates)} grid coordinates "
              f"(nx={nx}, ny={ny}, nz={nz}, dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f})")

    return coordinates



