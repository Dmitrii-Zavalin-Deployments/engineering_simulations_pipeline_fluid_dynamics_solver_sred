import numpy as np

def initialize_fields(num_nodes, initial_velocity, initial_pressure):
    """
    Initializes velocity and pressure fields for the entire domain.
    Assumes a uniform initial state based on inlet conditions.
    """
    velocity = np.full((num_nodes, 3), initial_velocity)
    pressure = np.full(num_nodes, initial_pressure)
    return velocity, pressure