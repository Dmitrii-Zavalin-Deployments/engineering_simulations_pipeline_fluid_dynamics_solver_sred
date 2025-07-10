# src/grid_generator.py

def generate_grid(domain: dict, initial_conditions: dict) -> list:
    """
    Stubbed grid generator. Each cell: [x, y, z, velocity_vector, pressure]
    """
    velocity = initial_conditions["initial_velocity"]
    pressure = initial_conditions["initial_pressure"]

    return [
        [0, 0, 0, velocity, pressure],
        [0, 1, 0, velocity, pressure],
        [1, 0, 0, velocity, pressure]
    ]



