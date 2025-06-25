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
    sim_instance.total_time = params.get('total_time', 1.0)
    sim_instance.time_step = params.get('time_step', 0.01)
    sim_instance.rho = params.get('density', 1.0)
    sim_instance.nu = params.get('kinematic_viscosity', 0.01)
    sim_instance.output_frequency_steps = params.get('output_frequency_steps', 10)
    # Set the solver type as a string, to be used for printing and instantiation.
    sim_instance.solver_type = params.get('solver_type', 'explicit')
    # Store initial velocity and pressure values for field initialization
    sim_instance.initial_velocity = input_data.get('initial_conditions', {}).get('velocity', [0.0, 0.0, 0.0])
    sim_instance.initial_pressure = input_data.get('initial_conditions', {}).get('pressure', 0.0)

def initialize_grid(sim_instance, input_data):
    """
    Initializes the grid based on input dimensions.
    """
    grid_data = input_data.get('grid', {})
    sim_instance.nx = grid_data.get('nx', 3)
    sim_instance.ny = grid_data.get('ny', 3)
    sim_instance.nz = grid_data.get('nz', 3)
    sim_instance.dx = grid_data.get('dx', 1.0)
    sim_instance.dz = grid_data.get('dz', 1.0)
    sim_instance.dy = grid_data.get('dy', 1.0) # Corrected to use dy
    
    # --- FIX 1: Populate mesh_info with all required data, including boundary conditions ---
    sim_instance.mesh_info = {
        'shape': (sim_instance.nx, sim_instance.ny, sim_instance.nz),
        'dx': sim_instance.dx,
        'dy': sim_instance.dy,
        'dz': sim_instance.dz,
        # This is the critical piece of missing information.
        'boundary_conditions': input_data.get('boundary_conditions', {})
    }
    
    print(f"Grid dimensions: {sim_instance.nx}x{sim_instance.ny}x{sim_instance.nz} cells")
    print(f"Grid spacing: dx={sim_instance.dx:.4f}, dy={sim_instance.dy:.4f}, dz={sim_instance.dz:.4f}")

def initialize_fields(sim_instance, input_data):
    """
    Initializes velocity and pressure fields with corrected shapes.
    """
    # --- FIX 2: Correctly size the arrays for a staggered grid with ghost cells ---
    # Velocity components are defined on cell faces, so they need one extra point per dimension.
    # We add 2 points for dimensions where nx/ny/nz > 1 for ghost cells.
    # For dimensions with size 1, we add 1 extra point for the boundary.
    u_shape = (sim_instance.nx + 2, sim_instance.ny + 2, sim_instance.nz + 2)
    v_shape = (sim_instance.nx + 2, sim_instance.ny + 2, sim_instance.nz + 2)
    w_shape = (sim_instance.nx + 2, sim_instance.ny + 2, sim_instance.nz + 2)
    p_shape = (sim_instance.nx + 2, sim_instance.ny + 2, sim_instance.nz + 2)
    
    # Initialize fields with zeros.
    sim_instance.u = np.full(u_shape, sim_instance.initial_velocity[0], dtype=np.float64)
    sim_instance.v = np.full(v_shape, sim_instance.initial_velocity[1], dtype=np.float64)
    sim_instance.w = np.full(w_shape, sim_instance.initial_velocity[2], dtype=np.float64)
    sim_instance.p = np.full(p_shape, sim_instance.initial_pressure, dtype=np.float64)

def print_initial_setup(sim_instance):
    """
    Prints a summary of the initial simulation setup.
    """
    print(f"Total time: {sim_instance.total_time:.2f} s, Time step: {sim_instance.time_step:.2f} s")
    print(f"Fluid: rho={sim_instance.rho}, nu={sim_instance.nu}")
    print(f"Solver: {sim_instance.solver_type.capitalize()}")
    print("-------------------------")
