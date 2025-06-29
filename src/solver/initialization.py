# src/solver/initialization.py

import json
import numpy as np

def load_input_data(input_file_path):
    """
    Loads pre-processed input data from a JSON file.
    """
    try:
        print(f"Loading pre-processed input from: {input_file_path}")
        with open(input_file_path, 'r') as f:
            input_data = json.load(f)
        return input_data
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {input_file_path} is not a valid JSON file.")
        raise

def initialize_simulation_parameters(sim_instance, input_data):
    """
    Initializes simulation parameters from the input data.
    """
    params = input_data.get('simulation_parameters', {})
    fluid = input_data.get('fluid_properties', {})
    ic = input_data.get('initial_conditions', {})

    sim_instance.total_time = params.get('total_time', 1.0)
    sim_instance.time_step = params.get('time_step', 0.01)
    sim_instance.output_frequency_steps = params.get('output_frequency_steps', 10)
    sim_instance.solver_type = params.get('solver_type', 'explicit')

    # ✅ Correctly extract from fluid_properties
    sim_instance.rho = fluid.get('density', 1.0)
    sim_instance.nu = fluid.get('viscosity', 0.01)

    # Allow backward compatibility with legacy keys
    sim_instance.initial_velocity = ic.get('initial_velocity') or ic.get('velocity') or [0.0, 0.0, 0.0]
    sim_instance.initial_pressure = ic.get('initial_pressure') or ic.get('pressure') or 0.0

def initialize_grid(sim_instance, input_data):
    """
    Initializes the grid based on preprocessed mesh_info section.
    """
    mesh_info = input_data.get('mesh_info', {})
    sim_instance.nx, sim_instance.ny, sim_instance.nz = mesh_info.get('grid_shape', [3, 3, 3])
    sim_instance.dx = mesh_info.get('dx', 1.0)
    sim_instance.dy = mesh_info.get('dy', 1.0)
    sim_instance.dz = mesh_info.get('dz', 1.0)

    sim_instance.mesh_info = dict(mesh_info)
    sim_instance.mesh_info['boundary_conditions'] = input_data.get('boundary_conditions', {})

    print(f"Grid dimensions: {sim_instance.nx}x{sim_instance.ny}x{sim_instance.nz} cells")
    print(f"Grid spacing: dx={sim_instance.dx:.4f}, dy={sim_instance.dy:.4f}, dz={sim_instance.dz:.4f}")

def initialize_fields(sim_instance, input_data):
    """
    Initializes velocity and pressure fields with appropriate ghost cell padding.
    """
    nx, ny, nz = sim_instance.nx + 2, sim_instance.ny + 2, sim_instance.nz + 2

    sim_instance.u = np.full((nx, ny, nz), sim_instance.initial_velocity[0], dtype=np.float64)
    sim_instance.v = np.full((nx, ny, nz), sim_instance.initial_velocity[1], dtype=np.float64)
    sim_instance.w = np.full((nx, ny, nz), sim_instance.initial_velocity[2], dtype=np.float64)
    sim_instance.p = np.full((nx, ny, nz), sim_instance.initial_pressure, dtype=np.float64)

def print_initial_setup(sim_instance):
    """
    Prints a summary of the initial simulation setup.
    """
    print(f"Total time: {sim_instance.total_time:.2f} s, Time step: {sim_instance.time_step:.2f} s")
    print(f"Fluid: rho={sim_instance.rho}, nu={sim_instance.nu}")
    print(f"Solver: {sim_instance.solver_type.capitalize()}")
    print("-------------------------")



