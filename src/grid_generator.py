# src/grid_generator.py

from src.grid_modules.cell import Cell
from grid_modules.grid_geometry import generate_coordinates
from grid_modules.initial_field_assigner import assign_fields
from grid_modules.boundary_manager import apply_boundaries

def generate_grid(domain: dict, initial_conditions: dict) -> list:
    """
    Generates a structured 3D grid with seeded fields and boundaries.
    Returns list of Cell objects.
    """
    coordinates = generate_coordinates(domain)

    # 🛡️ Fallback for missing or malformed initial conditions
    velocity = initial_conditions.get("initial_velocity")
    if not isinstance(velocity, list) or len(velocity) != 3:
        print("⚠️ initial_velocity missing or invalid. Using default [0.0, 0.0, 0.0].")
        velocity = [0.0, 0.0, 0.0]

    pressure = initial_conditions.get("initial_pressure")
    if not isinstance(pressure, (int, float)):
        print("⚠️ initial_pressure missing or invalid. Using default 0.0.")
        pressure = 0.0

    # Create cell objects using resolved initial values
    cells = [Cell(x, y, z, velocity=velocity, pressure=pressure) for (x, y, z) in coordinates]

    # Apply boundary conditions or ghost cells
    cells = apply_boundaries(cells, domain)

    return cells



