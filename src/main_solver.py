# src/main_solver.py

import json
import numpy as np
import time
import sys
import os

# --- Import Simulation Modules ---
try:
    from src.utils.grid import create_structured_grid_info, get_cell_centers
    from src.physics.boundary_conditions import apply_boundary_conditions, identify_boundary_nodes
    
    from src.numerical_methods.explicit_solver import solve_explicit
    from src.numerical_methods.implicit_solver import solve_implicit
    
    from src.utils.io import write_output_to_vtk
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
    
    # --- Grid Initialization ---
    grid_shape = (nx, ny, nz)
    num_nodes = nx * ny * nz # Total number of nodes/cells in the structured grid

    # Get cell centers for the structured grid. This array is used for mapping mesh faces to grid cells.
    # It will also be used in the final results output.
    nodes_coords = get_cell_centers(min_x, max_x, min_y, max_y, min_z, max_z, nx, ny, nz)

    # Initialize velocity and pressure fields
    velocity_field = np.zeros((*grid_shape, 3), dtype=np.float64) # Ensure float64 for numerical stability
    pressure_field = np.zeros(grid_shape, dtype=np.float64)

    # Prepare mesh_info dictionary for numerical methods and boundary conditions
    mesh_info = {
        'grid_shape': grid_shape,
        'dx': dx, 'dy': dy, 'dz': dz,
        'min_x': min_x, 'max_x': max_x,
        'min_y': min_y, 'max_y': max_y,
        'min_z': min_z, 'max_z': max_z,
        'all_cell_centers_flat': nodes_coords, # Provide cell centers for boundary identification
    }

    # Construct fluid_properties dictionary for apply_boundary_conditions
    fluid_properties_dict = {
        "density": density,
        "viscosity": viscosity
    }

    # --- Process Boundary Conditions ---
    # Extract mesh boundary faces from input_data. This is crucial for identifying grid cells from face_ids.
    all_mesh_boundary_faces = input_data.get('mesh', {}).get('boundary_faces', [])
    if not all_mesh_boundary_faces:
        print("Warning: No 'boundary_faces' found in the 'mesh' section of input data. "
              "Boundary conditions might not be applied correctly if they rely on mesh faces.", file=sys.stderr)

    # Process raw boundary conditions data into a more usable format
    # The `identify_boundary_nodes` function now takes `all_mesh_boundary_faces`
    processed_bcs = identify_boundary_nodes(boundary_conditions_data, all_mesh_boundary_faces, mesh_info)
    mesh_info['boundary_conditions'] = processed_bcs # Add processed BCs to mesh_info

    # --- Apply Initial Boundary Conditions ---
    # Apply initial boundary conditions to the fields
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

    # Pre-calculate grid lines for VTK output, as they are not stored in mesh_info anymore
    x_coords_grid_lines = np.linspace(min_x, max_x, nx + 1)
    y_coords_grid_lines = np.linspace(min_y, max_y, ny + 1)
    z_coords_grid_lines = np.linspace(min_z, max_z, nz + 1)


    for step in range(num_time_steps): # Loop through the number of total time steps
        current_time += time_step
        step_count += 1
        
        try:
            if solver_type == "explicit":
                velocity_field, pressure_field = solve_explicit(
                    velocity_field, pressure_field, density, viscosity,
                    dx, dy, dz, time_step, mesh_info # Pass mesh_info to solvers for BC application
                )
                # Re-apply boundary conditions after the explicit step to ensure consistency
                apply_boundary_conditions(
                    velocity_field, 
                    pressure_field, 
                    fluid_properties_dict, 
                    mesh_info, 
                    is_tentative_step=False # Applies to the final, updated fields after projection
                )

            elif solver_type == "implicit":
                velocity_field, pressure_field = solve_implicit(
                    velocity_field, pressure_field, density, viscosity,
                    dx, dy, dz, time_step, mesh_info # Pass mesh_info to solvers for BC application
                )
                # Re-apply boundary conditions after the implicit step to ensure consistency
                apply_boundary_conditions(
                    velocity_field, 
                    pressure_field, 
                    fluid_properties_dict, 
                    mesh_info, 
                    is_tentative_step=False # Applies to the final, updated fields after solving
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
            # Assuming output files go into a 'results' subdirectory relative to the script
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
