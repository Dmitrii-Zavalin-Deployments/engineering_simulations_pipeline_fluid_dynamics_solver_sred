import os
import numpy as np
import json

# Import modules from our new structure
from utils.grid import create_structured_grid_info, find_optimal_grid_dimensions
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
    
    # --- START OF CHANGES ---

    # Initialize domain min/max based on discovered boundary face nodes
    # This is crucial for matching the validator's expectations of the overall grid
    overall_domain_min = np.array([float('inf'), float('inf'), float('inf')])
    overall_domain_max = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    # Collect all unique node coordinates from boundary_faces to infer the overall domain
    # This loop is already present but ensure its purpose for domain inference is clear.
    found_boundary_nodes = False
    if "boundary_faces" in mesh_data and mesh_data["boundary_faces"]:
        for face_info in mesh_data["boundary_faces"]:
            for node_coords_str in face_info["nodes"].values():
                coords = np.array(node_coords_str)
                overall_domain_min = np.minimum(overall_domain_min, coords)
                overall_domain_max = np.maximum(overall_domain_max, coords)
                found_boundary_nodes = True

    # If no boundary faces with coordinates are provided, default to a unit cube for now.
    # In a real-world scenario, you might want to raise an error or have another input
    # method for domain definition if boundary faces don't fully define it.
    if not found_boundary_nodes:
        print("Warning: No boundary face coordinates found in input. Defaulting to a unit cube domain [0,0,0] to [1,1,1].")
        overall_domain_min = np.array([0.0, 0.0, 0.0])
        overall_domain_max = np.array([1.0, 1.0, 1.0])
    
    # Calculate domain size for grid dimension inference.
    # Ensure very small dimensions are treated as 0 for 2D/1D cases.
    domain_extent = overall_domain_max - overall_domain_min
    for i in range(3):
        if domain_extent[i] < 1e-9: # Use a small tolerance for floating point comparisons
            domain_extent[i] = 0.0
    domain_extent = tuple(domain_extent) # Convert to tuple as expected by find_optimal_grid_dimensions

    # Find optimal grid dimensions based on the total number of nodes and the inferred domain extent.
    # This function in utils/grid.py will be crucial for determining Nx, Ny, Nz.
    desired_grid_dims = find_optimal_grid_dimensions(num_nodes_from_json, domain_extent)
    print(f"Automatically determined grid dimensions: {desired_grid_dims}")
    
    # Now, create the structured grid information using the derived domain and dimensions.
    # The 'origin' parameter ensures the grid starts at overall_domain_min.
    num_nodes_actual, nodes_coords, grid_shape, dx, dy, dz, node_to_idx, idx_to_node = \
        create_structured_grid_info(grid_dims=desired_grid_dims, domain_size=domain_extent, origin=overall_domain_min)

    # Note: mesh_data["nodes"] already holds num_nodes_from_json.
    # num_nodes_actual from create_structured_grid_info should ideally be the same.
    # We use num_nodes_actual for consistency in the mesh_info dictionary.
    
    mesh_info = {
        "nodes": num_nodes_actual, # Use the actual number of nodes generated by grid.py
        "nodes_coords": nodes_coords, # This is the array of ALL node coordinates
        "grid_shape": grid_shape,
        "dx": dx, "dy": dy, "dz": dz,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node
    }

    # --- END OF CHANGES ---

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
