# src/grid_modules/grid_geometry.py

import logging

def generate_coordinates(domain: dict) -> list[tuple[float, float, float]]:
    """
    Generates 3D (x, y, z) physical coordinates for a structured grid,
    based on domain boundaries and resolution. Requires explicit bounds.

    Args:
        domain (dict): Must include keys:
            "min_x", "max_x", "nx",
            "min_y", "max_y", "ny",
            "min_z", "max_z", "nz"

    Returns:
        list of tuple: Each tuple represents (x, y, z) physical coordinate

    Raises:
        ValueError: If any required key is missing or resolution is invalid
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

    # Compute grid spacing
    dx = (max_x - min_x) / nx if nx > 0 else 0.0
    dy = (max_y - min_y) / ny if ny > 0 else 0.0
    dz = (max_z - min_z) / nz if nz > 0 else 0.0

    coordinates = []
    for i in range(nx):
        x = min_x + (i + 0.5) * dx
        for j in range(ny):
            y = min_y + (j + 0.5) * dy
            for k in range(nz):
                z = min_z + (k + 0.5) * dz
                coordinates.append((x, y, z))

    return coordinates



