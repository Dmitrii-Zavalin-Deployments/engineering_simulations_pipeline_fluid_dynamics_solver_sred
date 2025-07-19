# ✅ Final Updated Input Helper — Threshold Sensitivity Applied
# 📄 Full Path: tests/utils/test_input_factory.py

def make_input_data(resolution="very_low", time_step=0.5):
    """
    Constructs a valid simulation input_data dictionary tuned for mutation sensitivity.

    Args:
        resolution (str): Grid resolution label ("very_low", "low", "normal", "high")
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
            "min_x": 0.0, "max_x": 2.0, "nx": 2,
            "min_y": 0.0, "max_y": 1.0, "ny": 1,
            "min_z": 0.0, "max_z": 1.0, "nz": 1
        }
    }



