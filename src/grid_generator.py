# src/grid_generator.py

def generate_grid(domain: dict, initial_conditions: dict) -> list:
    """
    Generates a structured grid based on domain resolution.
    Each cell: [x, y, z, velocity_vector, pressure]
    Stubbed to return one layer slice (xy-plane at z=0)

    Args:
        domain (dict): Includes nx, ny, nz, and boundaries
        initial_conditions (dict): Includes velocity and pressure

    Returns:
        list: List of cell entries
    """
    nx, ny = domain["nx"], domain["ny"]
    velocity = initial_conditions["initial_velocity"]
    pressure = initial_conditions["initial_pressure"]

    grid = []

    for i in range(nx):
        for j in range(ny):
            cell = [i, j, 0, velocity, pressure]  # z=0 slice for simplicity
            grid.append(cell)

    return grid



