# ✅ Final Updated Input Helper
# 📄 Full Path: tests/utils/test_input_factory.py

def make_input_data(resolution="normal", time_step=0.05):
    """
    Constructs a valid simulation input_data dictionary with required domain parameters.

    Args:
        resolution (str): Grid resolution label ("normal", "high", "low")
        time_step (float): Simulation time step value

    Returns:
        dict: Complete input_data ready for simulation functions
    """
    return {
        "grid_resolution": resolution,
        "simulation_parameters": {
            "time_step": time_step
        },
        "domain_definition": {
            "min_x": 0.0,
            "max_x": 2.0,
            "nx": 2,
            "min_y": 0.0,
            "max_y": 1.0,
            "ny": 1
        }
    }



