# src/solver/initialization.py

import json
import numpy as np
import sys
import os

def load_input_data(filepath):
    """Loads and returns the pre-processed JSON data."""
    print(f"Loading pre-processed input from: {filepath}")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Input file not found at {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading input data: {e}", file=sys.stderr)
        sys.exit(1)


def initialize_simulation_parameters(simulation_instance, input_data):
    """Initializes simulation parameters from the input data."""
    sim_params = input_data.get('simulation_parameters', {})
    simulation_instance.time_step = sim_params.get('time_step')
    simulation_instance.total_time = sim_params.get('total_time')
    simulation_instance.solver = sim_params.get('solver', 'explicit')
    simulation_instance.output_frequency_steps = sim_params.get('output_frequency_steps', 1)

    fluid_props = input_data.get('fluid_properties', {})
    simulation_instance.rho = fluid_props.get('density')  # Density
    simulation_instance.nu = fluid_props.get('viscosity') # Viscosity


def initialize_grid(simulation_instance, input_data):
    """
    Initializes the simulation grid from 'mesh_info' or 'domain_settings'.
    
    This function is robust and checks for the 'mesh_info' key first, 
    falling back to 'domain_settings' if it's not present.
    """
    # We will prioritize 'mesh_info' because the pre-processing step creates it.
    mesh_info = input_data.get('mesh_info', {})
    
    # If mesh_info is empty, check the domain_settings key as a fallback.
    if not mesh_info:
        mesh_info = input_data.get('domain_settings', {})

    # Safely get the grid shape. Use a default value if the key is still missing.
    grid_shape = tuple(mesh_info.get('grid_shape', [1, 1, 1]))
    
    # Assign attributes to the simulation instance
    simulation_instance.mesh_info = mesh_info
    simulation_instance.grid_shape = grid_shape
    simulation_instance.nx, simulation_instance.ny, simulation_instance.nz = grid_shape
    
    simulation_instance.dx = mesh_info.get('dx')
    simulation_instance.dy = mesh_info.get('dy')
    simulation_instance.dz = mesh_info.get('dz')
    
    # Check if grid parameters were successfully loaded
    if not all([simulation_instance.nx, simulation_instance.ny, simulation_instance.nz,
                simulation_instance.dx, simulation_instance.dy, simulation_instance.dz]):
        raise ValueError("Grid parameters (nx, ny, nz, dx, dy, dz) could not be initialized. Check input JSON.")
    
    # Define grid arrays
    simulation_instance.x = np.linspace(mesh_info['min_x'], mesh_info['max_x'], simulation_instance.nx + 1)
    simulation_instance.y = np.linspace(mesh_info['min_y'], mesh_info['max_y'], simulation_instance.ny + 1)
    simulation_instance.z = np.linspace(mesh_info['min_z'], mesh_info['max_z'], simulation_instance.nz + 1)


def initialize_fields(simulation_instance, input_data):
    """Initializes velocity and pressure fields with initial conditions."""
    ic = input_data.get('initial_conditions', {})
    u_init, v_init, w_init = ic.get('initial_velocity', [0.0, 0.0, 0.0])
    p_init = ic.get('initial_pressure', 0.0)

    # Initialize fields with zeros, then apply initial conditions
    simulation_instance.u = np.full((simulation_instance.nx + 1, simulation_instance.ny + 1, simulation_instance.nz + 1), u_init)
    simulation_instance.v = np.full((simulation_instance.nx + 1, simulation_instance.ny + 1, simulation_instance.nz + 1), v_init)
    simulation_instance.w = np.full((simulation_instance.nx + 1, simulation_instance.ny + 1, simulation_instance.nz + 1), w_init)
    simulation_instance.p = np.full((simulation_instance.nx + 1, simulation_instance.ny + 1, simulation_instance.nz + 1), p_init)
    
    # Store a copy of the boundary conditions for access during the simulation loop
    simulation_instance.boundary_conditions = input_data.get('boundary_conditions', {})


def print_initial_setup(simulation_instance):
    """Prints the initial simulation setup for verification."""
    print(f"Grid dimensions: {simulation_instance.nx}x{simulation_instance.ny}x{simulation_instance.nz} cells")
    print(f"Grid spacing: dx={simulation_instance.dx:.4f}, dy={simulation_instance.dy:.4f}, dz={simulation_instance.dz:.4f}")
    print(f"Total time: {simulation_instance.total_time} s, Time step: {simulation_instance.time_step} s")
    print(f"Fluid: rho={simulation_instance.rho}, nu={simulation_instance.nu}")
    print(f"Solver: {simulation_instance.solver.capitalize()}")
    print("-" * 25)