# src/grid_modules/grid_geometry.py

def generate_coordinates(domain: dict) -> list[tuple[float, float, float]]:
    """
    Generates 3D (x, y, z) physical coordinates for a structured grid,
    based on domain boundaries and resolution.

    Args:
        domain (dict): Dictionary containing "min_x", "max_x", "nx",
                       "min_y", "max_y", "ny", "min_z", "max_z", "nz"

    Returns:
        list of tuple: Each tuple represents (x, y, z) physical coordinate
    """
    # Extract domain bounds and resolution
    min_x, max_x, nx = domain["min_x"], domain["max_x"], domain["nx"]
    min_y, max_y, ny = domain["min_y"], domain["max_y"], domain["ny"]
    min_z, max_z, nz = domain["min_z"], domain["max_z"], domain["nz"]

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



