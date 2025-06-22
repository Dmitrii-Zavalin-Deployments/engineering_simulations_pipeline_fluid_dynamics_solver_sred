import os
import numpy as np
import json

# Import modules from our new structure
# We will modify create_structured_grid_info to accept explicit axis coordinates
from utils.grid import create_structured_grid_info
# find_optimal_grid_dimensions might become obsolete or serve as a fallback,
# but we'll remove its direct usage for grid generation for now.
from utils.io import load_json
from physics.initialization import initialize_fields
from physics.boundary_conditions import apply_boundary_conditions
from numerical_methods.advection import compute_advection_acceleration
from numerical_methods.diffusion import compute_diffusion_acceleration
from numerical_methods.pressure_divergence import compute_velocity_divergence
from numerical_methods.poisson_solver import solve_poisson_for_phi
from numerical_methods.pressure_correction import correct_velocity_and_pressure

def compute_next_step(velocity, pressure, mesh_info, fluid_properties, dt):
    """
    Computes velocity and pressure updates using explicit Euler method
    and simplified finite difference approximations for Navier-Stokes terms.
    Applies a basic pressure projection method for incompressibility.
    """
    viscosity = fluid_properties["viscosity"]
    density = fluid_properties["density"]
    
    # Step 1: Compute tentative velocity (u* = u_n + dt * (adv + diff))
    advection_accel = compute_advection_acceleration(velocity, mesh_info)
    diffusion_accel = compute_diffusion_acceleration(velocity, mesh_info, viscosity)
    
    u_tentative = velocity + dt * (-advection_accel + diffusion_accel)

    # Step 2: Pressure Projection (Enforce Incompressibility)
    divergence_ut = compute_velocity_divergence(u_tentative, mesh_info)
    poisson_rhs = (density / dt) * divergence_ut
    phi = solve_poisson_for_phi(poisson_rhs, mesh_info)

    # Step 3: Correct Velocity and Update Pressure
    velocity, pressure = correct_velocity_and_pressure(u_tentative, pressure, phi, mesh_info, dt, density)
    
    return velocity, pressure


def run_simulation(json_filename):
    """
    Runs the fluid simulation based on simplified Navier-Stokes equations.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming data/testing-input-output is relative to src/
    data_dir = os.path.join(current_dir, "..", "data", "testing-input-output")
    
    input_filepath = os.path.join(data_dir, json_filename)
    data = load_json(input_filepath)

    mesh_data = data["mesh"]
    fluid_properties = data["fluid_properties"]
    boundary_conditions = data["boundary_conditions"]
    simulation_params = data["simulation_parameters"]

    time_step = simulation_params["time_step"]
    total_time = simulation_params["total_time"]
    num_steps = int(total_time / time_step)

    num_nodes_from_json = mesh_data["nodes"]
    
    # --- START OF MAJOR CHANGES ---

    # Collect all unique X, Y, Z coordinates from the boundary faces.
    # This assumes the boundary faces implicitly define the structured grid lines.
    all_x_coords = set()
    all_y_coords = set()
    all_z_coords = set()
    
    found_boundary_nodes = False
    if "boundary_faces" in mesh_data and mesh_data["boundary_faces"]:
        for face_info in mesh_data["boundary_faces"]:
            for node_coords_str in face_info["nodes"].values():
                coords = np.array(node_coords_str)
                all_x_coords.add(coords[0])
                all_y_coords.add(coords[1])
                all_z_coords.add(coords[2])
                found_boundary_nodes = True
    
    # Sort the unique coordinates to define the grid lines explicitly
    # This is critical for matching the validator's expected node order and positions.
    x_coords_grid_lines = sorted(list(all_x_coords))
    y_coords_grid_lines = sorted(list(all_y_coords))
    z_coords_grid_lines = sorted(list(all_z_coords))

    # Determine grid dimensions (Nx, Ny, Nz) directly from the number of unique coordinates
    nx = len(x_coords_grid_lines)
    ny = len(y_coords_grid_lines)
    nz = len(z_coords_grid_lines)

    # If no boundary nodes were found, or if dimensions are 0, fall back to a default,
    # though this scenario might indicate an invalid input for a structured grid.
    if not found_boundary_nodes or nx * ny * nz == 0:
        print("Warning: Could not infer structured grid from boundary faces. Defaulting to a unit cube with sqrt(num_nodes) dimensions.")
        # Fallback for truly undefined boundary coords - this might still cause mismatches
        # if the validator expects something specific without boundary faces.
        # This part of the logic might need further refinement based on specific input file expectations.
        # For a truly structured grid, boundary_faces should always define the extent.
        side_len = round(num_nodes_from_json**(1/3))
        if side_len == 0: side_len = 1
        nx, ny, nz = side_len, side_len, side_len
        if nx*ny*nz != num_nodes_from_json: # Simple cubic division might not work for all num_nodes
            # Fallback to a single axis for simplicity if the cubic approximation fails
            # This is a very rough fallback and likely won't match arbitrary validator expectations
            if num_nodes_from_json > 0:
                nx, ny, nz = num_nodes_from_json, 1, 1
            else:
                nx, ny, nz = 1,1,1 # Minimal case
        
        # Define default domain for the fallback case
        x_coords_grid_lines = np.linspace(0.0, 1.0, nx).tolist()
        y_coords_grid_lines = np.linspace(0.0, 1.0, ny).tolist()
        z_coords_grid_lines = np.linspace(0.0, 1.0, nz).tolist()

    # Verify that the derived total number of nodes matches the input schema's declared node count
    actual_derived_nodes = nx * ny * nz
    if actual_derived_nodes != num_nodes_from_json:
        print(f"⚠️ Warning: Inferred grid (Nx={nx}, Ny={ny}, Nz={nz} = {actual_derived_nodes} nodes) "
              f"does not match declared mesh.nodes ({num_nodes_from_json}). Using inferred grid dimensions.")
        # It's important to use the *inferred* grid if it's the one we are constructing,
        # but this indicates a potential inconsistency in the input JSON itself.
        # For now, we proceed with the derived (nx, ny, nz) which define the actual grid generated.

    # Now, create the structured grid information using the derived axis coordinates
    num_nodes_actual, nodes_coords, grid_shape, dx, dy, dz, node_to_idx, idx_to_node = \
        create_structured_grid_info(x_coords_grid_lines, y_coords_grid_lines, z_coords_grid_lines)

    mesh_info = {
        "nodes": num_nodes_actual, # This should now be nx*ny*nz from actual grid lines
        "nodes_coords": nodes_coords, # This is the array of ALL node coordinates
        "grid_shape": grid_shape,    # This will be (nx, ny, nz)
        "dx": dx, "dy": dy, "dz": dz,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node
    }

    # --- END OF MAJOR CHANGES ---

    initial_velocity = np.array(boundary_conditions["inlet"]["velocity"])
    initial_pressure = boundary_conditions["inlet"]["pressure"]
    velocity, pressure = initialize_fields(mesh_info["nodes"], initial_velocity, initial_pressure)

    # CFL calculation and warning (existing logic)
    max_inlet_velocity_magnitude = np.linalg.norm(initial_velocity)
    if max_inlet_velocity_magnitude > 1e-9:
        finite_spacings = [s for s in [dx, dy, dz] if s > 1e-9]
        if not finite_spacings:
            cfl_dt_limit = float('inf')
        else:
            cfl_dt_limit = min(finite_spacings) / max_inlet_velocity_magnitude
            
        target_cfl_num = 0.5    
        recommended_dt = target_cfl_num * cfl_dt_limit

        if time_step > recommended_dt:
            print(f"⚠️ Warning: Time step ({time_step:.4f}s) exceeds recommended CFL limit ({recommended_dt:.4f}s).")
            print(f"    This may lead to numerical instability. Consider reducing time_step.")
    else:
        print("Note: Cannot calculate CFL limit (inlet velocity is zero). Time step stability not checked.")

    all_velocities = []
    all_pressures = []
    time_points = []

    print(f"Simulation starting for {num_steps} steps (Total Time: {total_time}s, Time Step: {time_step}s)")
    print(f"Derived 3D grid shape: {grid_shape} with dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    print(f"Total nodes: {mesh_info['nodes']}")

    for step in range(num_steps):
        current_time = (step + 1) * time_step
        
        apply_boundary_conditions(velocity, pressure, boundary_conditions, mesh_info)
        
        velocity, pressure = compute_next_step(velocity, pressure, mesh_info, fluid_properties, time_step)
        
        all_velocities.append(velocity.tolist())
        all_pressures.append(pressure.tolist())
        time_points.append(current_time)

        if (step + 1) % 10 == 0 or step == num_steps - 1:
            print(f"Step {step+1}/{num_steps} (Time: {current_time:.2f}s): "
                  f"Avg Velocity = {np.mean(velocity, axis=0)}, Avg Pressure = {np.mean(pressure):.2f}")

    output_filename = os.path.join(data_dir, "navier_stokes_results.json")
    with open(output_filename, "w") as output_file:
        json.dump({
            "time_points": time_points,
            "velocity_history": all_velocities,
            "pressure_history": all_pressures,
            "mesh_info": {
                "nodes": mesh_info["nodes"],
                "nodes_coords": mesh_info["nodes_coords"].tolist(), # Convert numpy array to list for JSON
                "grid_shape": mesh_info["grid_shape"],
                "dx": mesh_info["dx"], "dy": mesh_info["dy"], "dz": mesh_info["dz"]
            }
        }, output_file, indent=4)

    print(f"✅ Simulation completed. Results saved to {output_filename}")

if __name__ == "__main__":
    # Ensure the input JSON file exists in data/testing-input-output/ relative to src/
    run_simulation("fluid_simulation_input.json")
