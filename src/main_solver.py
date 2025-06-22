# src/main_solver.py

import json
import numpy as np
import time
import sys
import os

# --- Import Simulation Modules ---
# These imports are now updated to reflect your directory structure
try:
    # UPDATED: Changed 'create_grid' to 'create_structured_grid_info'
    # NOTE: The create_structured_grid_info function is in src/utils/grid.py
    # but based on your project structure, it seems the main_solver.py might also
    # need access to `get_cell_centers` from grid.py, which I added in a previous step.
    # So, we keep both here.
    from src.utils.grid import create_structured_grid_info, get_cell_centers
    from src.physics.boundary_conditions import apply_boundary_conditions, identify_boundary_nodes # Added identify_boundary_nodes here
    
    # UPDATED IMPORTS: Pointing to the new solver files in numerical_methods
    from src.numerical_methods.explicit_solver import solve_explicit
    from src.numerical_methods.implicit_solver import solve_implicit
    
    # If you have other physics functions (e.g., for calculating forces, etc.)
    # from src.physics.navier_stokes import calculate_velocities_pressures 
    # Also need output_writer
    from src.utils.io import write_output_to_vtk # Assuming this function is in src/utils/io.py
except ImportError as e:
    print(f"Error importing essential simulation modules: {e}", file=sys.stderr)
    print("Please ensure all modules and functions (e.g., 'src/utils/grid.py', "
          "'src/physics/boundary_conditions.py', and 'src/numerical_methods/solver_files') "
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
    # Removed the consistency check warnings here as pre_process_input.py handles this
    # by generating a uniform grid from potentially non-uniform input extents.
    # The dx, dy, dz values are now *derived* from the extents and number of unique planes.


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
    output_interval = simulation_parameters.get('output_interval', 1) # Default to 1 if not specified
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

    # This call is important for the `mesh_info.nodes_coords` in the output
    nodes_coords = get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz)

    # Initialize velocity and pressure fields
    # Velocity field is typically (nx, ny, nz, 3) for 3 components at each grid point (cell center)
    # Pressure field is (nx, ny, nz)
    velocity_field = np.zeros((*grid_shape, 3))
    pressure_field = np.zeros(grid_shape)

    # Prepare mesh_info dictionary for numerical methods and boundary conditions
    # This also needs to store the grid lines for identify_boundary_nodes
    x_coords_grid_lines = np.linspace(min_x, max_x, nx + 1)
    y_coords_grid_lines = np.linspace(min_y, max_y, ny + 1)
    z_coords_grid_lines = np.linspace(min_z, max_z, nz + 1)

    mesh_info = {
        'grid_shape': grid_shape,
        'dx': dx, 'dy': dy, 'dz': dz,
        'x_coords_grid_lines': x_coords_grid_lines,
        'y_coords_grid_lines': y_coords_grid_lines,
        'z_coords_grid_lines': z_coords_grid_lines,
        # 'boundary_conditions' will be added after processing
    }

    # --- Process Boundary Conditions ---
    # Process raw boundary conditions data into a more usable format
    processed_bcs = identify_boundary_nodes(boundary_conditions_data, mesh_info)
    mesh_info['boundary_conditions'] = processed_bcs # Add processed BCs to mesh_info

    # Construct fluid_properties dictionary for apply_boundary_conditions
    fluid_properties_dict = {
        "density": density,
        "viscosity": viscosity
    }

    # --- Apply Initial Boundary Conditions ---
    # Apply initial boundary conditions to the fields
    # The function modifies fields in-place, so no assignment is needed.
    apply_boundary_conditions(
        velocity_field, 
        pressure_field, 
        fluid_properties_dict, # Pass as a dictionary
        mesh_info,             # This dict now contains processed BCs and grid info
        is_tentative_step=False # Applying to initial, final fields
    )

    # --- Simulation Loop ---
    num_time_steps = int(total_time / time_step)
    current_time = 0.0
    step_count = 0

    print(f"\n--- Starting Fluid Simulation ---")
    print(f"Grid Dimensions: {nx}x{ny}x{nz} cells")
    print(f"Grid Spacing (dx, dy, dz): ({dx:.4e}, {dy:.4e}, {dz:.4e})")
    print(f"Fluid Properties: Density={density} kg/m^3, Viscosity={viscosity} Pa.s")
    print(f"Simulation Parameters: Time Step={time_step} s, Total Time={total_time} s, Solver={solver_type}")
    print(f"---------------------------------\n")


    start_sim_time = time.time()

    for step in range(num_time_steps): # Loop through the number of total time steps
        current_time += time_step
        step_count += 1
        
        # Apply boundary conditions at the beginning of each step (before solving)
        # This function modifies velocity_field and pressure_field in place or returns new arrays
        # Note: Apply boundary conditions to tentative velocity (u_star) when relevant in solvers.
        # This main loop call applies to the overall velocity_field before the next iteration's solve.
        # Specific BCs for u_star might be handled within explicit_solver/implicit_solver
        # if the solver uses a projection method.
        
        # Choose solver based on input parameters
        try:
            if solver_type == "explicit":
                velocity_field, pressure_field = solve_explicit(
                    velocity_field, pressure_field, density, viscosity,
                    dx, dy, dz, time_step
                )
                # Re-apply boundary conditions to the final velocity/pressure after the explicit step
                # This is crucial for maintaining boundary conditions after the solve.
                apply_boundary_conditions(
                    velocity_field, 
                    pressure_field, 
                    fluid_properties_dict, 
                    mesh_info, 
                    is_tentative_step=False # This applies to the final, updated fields
                )

            elif solver_type == "implicit":
                velocity_field, pressure_field = solve_implicit(
                    velocity_field, pressure_field, density, viscosity,
                    dx, dy, dz, time_step
                )
                # Re-apply boundary conditions to the final velocity/pressure after the implicit step
                apply_boundary_conditions(
                    velocity_field, 
                    pressure_field, 
                    fluid_properties_dict, 
                    mesh_info, 
                    is_tentative_step=False # This applies to the final, updated fields
                )
            else:
                raise ValueError(f"Unsupported solver type specified in simulation_parameters: '{solver_type}'. "
                                 "Choose 'explicit' or 'implicit'.")
        except Exception as e:
            print(f"An error occurred during time step {step_count}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1) # Exit immediately on solver error

        if step_count % 10 == 0: # Print progress every 10 steps
            print(f"Simulation Time: {current_time:.4f}s / {total_time:.4f}s ({step_count} steps)")
            
        # Output results at specified intervals
        if (step_count % output_interval == 0) or (step_count == num_time_steps):
            output_filename = f"navier_stokes_results_step_{step_count}.vti"
            # Assuming output files go into a 'results' subdirectory in GITHUB_WORKSPACE
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(output_dir, exist_ok=True)
            output_filepath = os.path.join(output_dir, output_filename)
            
            print(f"Writing output to {output_filepath} at time {current_time:.4f}s")
            write_output_to_vtk(
                velocity_field,
                pressure_field,
                x_coords_grid_lines, y_coords_grid_lines, z_coords_grid_lines,
                output_filepath
            )


    end_sim_time = time.time()
    print(f"\n--- Simulation Complete ---")
    print(f"Total simulation time: {end_sim_time - start_sim_time:.2f} seconds.")
    print(f"---------------------------\n")


    # --- Prepare Results for Output Schema (navier_stokes_results.schema.json) ---
    # For now, we are writing VTI files directly. If a final summary JSON is needed:
    results = {
        "time_points": [current_time], # Only final time
        "velocity_history": [velocity_field.reshape(-1, 3).tolist()], # Only final state
        "pressure_history": [pressure_field.flatten().tolist()], # Only final state
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
    output_file = sys.argv[2] # This will be the main navier_stokes_results.json, not the VTK files

    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: '{input_file}'")

        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Pass the desired output directory for VTK files via input_data or directly
        # The output_file (navier_stokes_output.json) will contain summary, VTK files will be detailed
        # For VTK, assuming a 'results' folder within the top-level project directory.
        
        simulation_results = run_simulation(input_data)

        # Ensure output directory for the final JSON result exists
        output_dir_json = os.path.dirname(output_file)
        if output_dir_json and not os.path.exists(output_dir_json):
            os.makedirs(output_dir_json)

        with open(output_file, 'w') as f:
            json.dump(simulation_results, f, indent=2)
        print(f"Summary simulation results successfully saved to '{output_file}'")

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
