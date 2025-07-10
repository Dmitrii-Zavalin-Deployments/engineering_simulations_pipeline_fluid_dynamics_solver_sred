# src/grid_modules/grid_geometry.py

def generate_coordinates(domain: dict) -> list:
    """
    Returns list of 3D (x, y, z) coordinates spaced by domain resolution.
    Stub: just uses index space for now.
    """
    nx, ny, nz = domain["nx"], domain["ny"], domain["nz"]
    coords = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                coords.append((i, j, k))  # Later: physical spacing

    return coords



