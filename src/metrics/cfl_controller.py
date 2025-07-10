# src/metrics/cfl_controller.py

def compute_global_cfl(grid: list, time_step: float) -> float:
    """
    Stub CFL calculator using velocity and time step.
    """
    velocity = grid[0][3][0] if grid else 0.0
    return velocity * time_step * 9.0



