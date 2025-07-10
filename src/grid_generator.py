# src/grid_generator.py

from grid_modules.cell import Cell
from grid_modules.grid_geometry import generate_coordinates
from grid_modules.initial_field_assigner import assign_fields
from grid_modules.boundary_manager import apply_boundaries

def generate_grid(domain: dict, initial_conditions: dict) -> list:
    """
    Generates a structured 3D grid with seeded fields and boundaries.
    Returns list of Cell objects.
    """
    coordinates = generate_coordinates(domain)

    # Create raw cell objects with placeholders
    cells = [Cell(x, y, z, velocity=[], pressure=0.0) for (x, y, z) in coordinates]

    # Assign initial velocity and pressure
    cells = assign_fields(cells, initial_conditions)

    # Apply boundary conditions or ghost cells
    cells = apply_boundaries(cells, domain)

    return cells



