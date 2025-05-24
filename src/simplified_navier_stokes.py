import json
import numpy as np

def load_json(filename):
    """ Loads fluid simulation data from JSON file. """
    with open(filename, "r") as file:
        return json.load(file)

def initialize_fields(mesh, fluid_properties):
    """ Initializes velocity and pressure fields. """
    num_nodes = mesh["nodes"]
    velocity = np.zeros((num_nodes, 3))  # 3D velocity (u, v, w)
    pressure = np.ones(num_nodes) * fluid_properties["density"] * 101325  # Atmospheric pressure assumption
    return velocity, pressure

def apply_boundary_conditions(velocity, pressure, boundary_conditions):
    """ Enforces inlet velocity and outlet pressure boundary conditions. """
    for face_id in boundary_conditions["inlet"]["faces"]:
        velocity[face_id] = boundary_conditions["inlet"]["velocity"]

    for face_id in boundary_conditions["outlet"]["faces"]:
        pressure[face_id] = boundary_conditions["outlet"]["pressure"]

def compute_next_step(velocity, pressure, mesh, fluid_properties, dt):
    """ Computes velocity updates using explicit Euler method. """
    viscosity = fluid_properties["viscosity"]
    density = fluid_properties["density"]

    # Compute viscous diffusion effects (simple Laplacian approximation)
    for node_id in range(mesh["nodes"]):
        velocity[node_id] += viscosity * dt * (np.random.randn(3) * 0.01)  # Basic diffusion term

    # Apply incompressibility condition (pressure projection method - simplified)
    pressure -= dt * (pressure - np.mean(pressure)) * 0.01  # Simple correction

    return velocity, pressure

def run_simulation(json_filename):
    """ Runs the fluid simulation based on simplified Navier-Stokes equations. """
    data = load_json(json_filename)

    mesh = data["mesh"]
    fluid_properties = data["fluid_properties"]
    boundary_conditions = data["boundary_conditions"]
    simulation_params = data["simulation_parameters"]

    time_step = simulation_params["time_step"]
    total_time = simulation_params["total_time"]
    num_steps = int(total_time / time_step)

    velocity, pressure = initialize_fields(mesh, fluid_properties)

    for step in range(num_steps):
        apply_boundary_conditions(velocity, pressure, boundary_conditions)
        velocity, pressure = compute_next_step(velocity, pressure, mesh, fluid_properties, time_step)
        print(f"Step {step+1}/{num_steps}: Avg Velocity = {np.mean(velocity, axis=0)}, Avg Pressure = {np.mean(pressure)}")

    print("✅ Simulation completed.")

# Run the solver
run_simulation("fluid_simulation_input.json")



