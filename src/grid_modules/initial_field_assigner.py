# src/grid_modules/initial_field_assigner.py

def assign_fields(cells: list, initial_conditions: dict) -> list:
    """
    Stub that assigns initial velocity and pressure to each cell.
    """
    velocity = initial_conditions["initial_velocity"]
    pressure = initial_conditions["initial_pressure"]

    for cell in cells:
        cell.velocity = velocity
        cell.pressure = pressure

    return cells



