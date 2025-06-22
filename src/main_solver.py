# src/main_solver.py

import json
import numpy as np
import time
import sys
import os

# --- Import Simulation Modules ---
# These imports are now updated to reflect your directory structure
try:
    from src.utils.grid import create_grid, get_cell_centers
    from src.physics.boundary_conditions import apply_boundary_conditions
    
    # UPDATED IMPORTS: Pointing to the new solver files in numerical_methods
    from src.numerical_methods.explicit_solver import solve_explicit
    from src.numerical_methods.implicit_solver import solve_implicit
    
    # If you have other physics functions (e.g., for calculating forces, etc.)
    # from src.physics.navier_stokes import calculate_velocities_pressures 
except ImportError as e:
    print(f"Error importing essential simulation modules: {e}", file=sys.stderr)
    print("Please ensure 'src/utils/grid.py', 'src/physics/boundary_conditions.py', "
          "and 'src/numerical_methods/explicit_solver.py'/'implicit_solver.py' "
          "contain the necessary functions and paths are correct.", file=sys.stderr)
    sys.exit(1)


def run_simulation(input_data: dict) -> dict:
    """
    Runs the fluid simulation based on the provided input data (pre-processed format).

    Args:
        input_data (dict): Pre-processed simulation input parameters.

    Returns:
        dict: Simulation results conforming to navier_stokes_results.schema.json.
    
    Raises:
        ValueError: If required input data is missing or invalid.
    """
    
    # --- Parse Pre-processed Input Data ---
    
    domain_settings = input_data.get("domain_settings")
    if not domain_settings:
        raise ValueError("Input JSON must contain 'domain_settings' section from pre-processing.")

    # Validate and extract domain parameters
    required_domain_params = ["min_x", "max_x", "min_y", "max_y", "min_z", "max_z", "dx", "dy", "dz", "nx", "ny", "nz"]
    for param in required_domain_params:
        if param not in domain_settings:
            raise ValueError(f"Missing required domain setting: '{param}' in pre-processed input.")
    
    min_x, max_x = domain_settings["min_x"], domain_settings["max_x"]
    min_y, max_y = domain_settings["min_y"], domain_settings["max_y"]
    min_z, max_z = domain_settings["min_z"], domain_settings["max_z"]
    dx, dy, dz = domain_settings["dx"], domain_settings["dy"], domain_settings["dz"]
    nx, ny, nz = domain_settings["nx"], domain_settings["ny"], domain_settings["nz"]

    # Basic validation for grid dimensions
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"Invalid grid dimensions: nx={nx}, ny={ny}, nz={nz}. Must be positive integers.")
    # Check if dx/dy/dz are consistent with extents and nx/ny/nz, allowing for 2D/1D cases (dx/dy/dz = 0.0)
    # A small tolerance is used for floating point comparison
    TOLERANCE = 1e-9
    if dx > TOLERANCE and abs((max_x - min_x) / dx - nx) > TOLERANCE and nx != 1:
        print(f"Warning: X-dimension consistency check failed. ((max_x - min_x) / dx) is {((max_x - min_x) / dx):.4e}, nx is {nx}.", file=sys.stderr)
    if dy > TOLERANCE and abs((max_y - min_y) / dy - ny) > TOLERANCE and ny != 1:
        print(f"Warning: Y-dimension consistency check failed. ((max_y - min_y) / dy) is {((max_y - min_y) / dy):.4e}, ny is {ny}.", file=sys.stderr)
    if dz > TOLERANCE and abs((max_z - min_z) / dz - nz) > TOLERANCE and nz != 1:
        print(f"Warning: Z-dimension consistency check failed. ((max_z - min_z) / dz) is {((max_z - min_z) / dz):.4e}, nz is {nz}.", file=sys.stderr)


    fluid_properties = input_data.get("fluid_properties")
    if not fluid_properties:
        raise ValueError("Input JSON must contain 'fluid_properties'.")
    density = fluid_properties.get("density")
    viscosity = fluid_properties.get("viscosity")
    if density is None or viscosity is None:
        raise ValueError("Missing 'density' or 'viscosity' in 'fluid_properties'.")

    simulation_parameters = input_data.get("simulation_parameters")
    if not simulation_parameters:
        raise ValueError("Input JSON must contain 'simulation_parameters'.")
    time_step = simulation_parameters.get("time_step")
    total_time = simulation_parameters.get("total_time")
    solver_type = simulation_parameters.get("solver")
    if time_step is None or total_time is None or solver_type is None:
        raise ValueError("Missing 'time_step', 'total_time', or 'solver' in 'simulation_parameters'.")

    boundary_conditions_data = input_data.get("boundary_conditions")
    if not boundary_conditions_data:
        raise ValueError("Input JSON must contain 'boundary_conditions' section.")
    # 'boundary_conditions_data' now directly contains the flags like 'min_x: true'
    # and the associated physical values, as prepared by pre_process_input.py.

    # --- Grid Initialization ---
    grid_shape = (nx, ny, nz)
    num_nodes = nx * ny * nz # Total number of nodes/cells in the structured grid

    # It's better to use create_grid for the internal grid setup (e.g. cell face areas, etc.)
    # and get_cell_centers to get the coordinates of where results are sampled for output.
    
    # This call is important for the `mesh_info.nodes_coords` in the output
    nodes_coords = get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz)


    # Initial conditions (e.g., zero velocity, zero pressure)
    # Use numpy arrays with correct shapes for vectorized operations
    # Velocity field is typically (nx, ny, nz, 3) for 3 components at each grid point (cell center)
    # Pressure field is (nx, ny, nz)
    velocity_field = np.zeros((*grid_shape, 3))
    pressure_field = np.zeros(grid_shape)

    # --- Simulation Loop ---
    time_points = []
    velocity_history = []
    pressure_history = []

    current_time = 0.0
    step_count = 0

    print(f"\n--- Starting Fluid Simulation ---")
    print(f"Grid Dimensions: {nx}x{ny}x{nz} cells")
    print(f"Grid Spacing (dx, dy, dz): ({dx:.4e}, {dy:.4e}, {dz:.4e})")
    print(f"Fluid Properties: Density={density} kg/m^3, Viscosity={viscosity} Pa.s")
    print(f"Simulation Parameters: Time Step={time_step} s, Total Time={total_time} s, Solver={solver_type}")
    print(f"---------------------------------\n")


    start_sim_time = time.time()

    while current_time <= total_time + TOLERANCE: # Use tolerance to ensure final time step is included
        # Store current state for history
        time_points.append(current_time)
        # Reshape for output: (num_nodes, 3) for velocity, (num_nodes,) for pressure
        velocity_history.append(velocity_field.reshape(-1, 3).tolist())
        pressure_history.append(pressure_field.flatten().tolist())

        # Apply boundary conditions at the beginning of each step
        # This function modifies velocity_field and pressure_field in place or returns new arrays
        velocity_field, pressure_field = apply_boundary_conditions(
            velocity_field, pressure_field, grid_shape,
            min_x, max_x, min_y, max_y, min_z, max_z,
            boundary_conditions_data # This is the pre-processed BC data
        )

        # Choose solver based on input parameters
        if solver_type == "explicit":
            velocity_field, pressure_field = solve_explicit(
                velocity_field, pressure_field, density, viscosity,
                dx, dy, dz, time_step
            )
        elif solver_type == "implicit":
            velocity_field, pressure_field = solve_implicit(
                velocity_field, pressure_field, density, viscosity,
                dx, dy, dz, time_step
            )
        else:
            raise ValueError(f"Unsupported solver type specified in simulation_parameters: '{solver_type}'. "
                             "Choose 'explicit' or 'implicit'.")

        current_time += time_step
        step_count += 1
        if step_count % 10 == 0: # Print progress every 10 steps
            print(f"Simulation Time: {current_time:.4f}s / {total_time:.4f}s ({step_count} steps)")
            
    end_sim_time = time.time()
    print(f"\n--- Simulation Complete ---")
    print(f"Total simulation time: {end_sim_time - start_sim_time:.2f} seconds.")
    print(f"---------------------------\n")


    # --- Prepare Results for Output Schema (navier_stokes_results.schema.json) ---
    results = {
        "time_points": time_points,
        "velocity_history": velocity_history,
        "pressure_history": pressure_history,
        "mesh_info": {
            "nodes": num_nodes, # Total number of nodes in the structured grid
            "nodes_coords": nodes_coords.reshape(-1, 3).tolist(), # Flatten to a list of [x,y,z] lists
            "grid_shape": list(grid_shape), # Ensure it's a list for JSON serialization
            "dx": dx,
            "dy": dy,
            "dz": dz
        }
    }
    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/main_solver.py <input_temp_json_filepath> <output_results_json_filepath>")
        print("Example: python src/main_solver.py temp/solver_input.json results/navier_stokes_output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: '{input_file}'")

        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        simulation_results = run_simulation(input_data)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_file, 'w') as f:
            json.dump(simulation_results, f, indent=2)
        print(f"Simulation results successfully saved to '{output_file}'")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration or Logic Error: {e}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"File I/O Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print full traceback for unexpected errors
        sys.exit(1)