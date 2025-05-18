import json
import numpy as np
from pint import UnitRegistry

# Initialize unit registry for scientific units
ureg = UnitRegistry()

# Load input file
def load_input_file(file_path):
    """Reads and validates the input JSON file with fluid properties."""
    with open(file_path, "r") as file:
        input_data = json.load(file)

    # Extract fluid properties from the inlet boundary
    fluid_properties = input_data["inlet_boundary"]["fluid_properties"]

    # Convert velocity into a NumPy array representing a vector
    input_data["fluid_velocity"] = np.array(input_data["inlet_boundary"]["velocity"], dtype=float) * ureg.meter / ureg.second

    # Convert units correctly
    input_data["pressure"] = input_data["inlet_boundary"]["pressure"] * ureg.pascal if input_data["inlet_boundary"]["pressure"] is not None else 0 * ureg.pascal
    input_data["density"] = fluid_properties["density"] * ureg.kilogram / ureg.meter**3
    input_data["viscosity"] = fluid_properties["viscosity"] * ureg.pascal * ureg.second

    return input_data

# Apply boundary conditions
def apply_boundary_conditions(mesh, input_data):
    """Assigns inlet, outlet, and wall boundary conditions."""
    mesh["boundary"]["inlet"]["velocity"] = input_data["fluid_velocity"]
    mesh["boundary"]["outlet"]["pressure"] = input_data["pressure"]
    mesh["boundary"]["walls"]["velocity"] = np.zeros(3, dtype=float) * ureg.meter / ureg.second  # No-slip condition
    return mesh

# CFL condition enforcement
def enforce_cfl_condition(velocity_field, dx_field, dt):
    """Ensures CFL condition across computational domain."""
    max_velocity = np.max(np.linalg.norm(velocity_field, axis=-1))
    min_dx = np.min(dx_field)
    cfl_value = max_velocity * dt / min_dx
    assert cfl_value <= 1, "CFL condition violated! Adjust time-step or grid spacing."

# Solve Navier-Stokes Equations
def solve_navier_stokes(input_data, grid_size=(100, 100, 3), dt=0.001):
    """Numerically solves Navier-Stokes using Finite Volume Method."""
    velocity = np.full(grid_size, input_data["fluid_velocity"], dtype=float)
    pressure = np.full(grid_size[:2], input_data["pressure"].magnitude, dtype=float)
    density = input_data["density"].magnitude
    viscosity = input_data["viscosity"].magnitude

    # Compute advection, diffusion, and pressure gradient separately for vx, vy, vz
    for _ in range(500):
        advection_term = -velocity * np.gradient(velocity, axis=(0, 1))
        diffusion_term = viscosity * np.array([np.gradient(np.gradient(velocity[..., i], axis=(0, 1)), axis=(0, 1)) for i in range(3)], dtype=float).T
        
        # Fix pressure gradient computation
        grad_x, grad_y = np.gradient(pressure, axis=(0, 1))  # Compute separate gradients
        pressure_gradient = -np.stack([grad_x, grad_y], axis=-1) / density  # Correct vector formatting
        
        velocity += dt * (advection_term + diffusion_term + pressure_gradient)

    return {"velocity": velocity.tolist(), "pressure": pressure.tolist()}

# Compute Turbulence using RANS k-epsilon Model
def compute_turbulence_rans(velocity_field, viscosity):
    """Computes turbulence dissipation rate using k-epsilon model."""
    velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
    k = 0.5 * np.mean(velocity_magnitude**2)
    epsilon = viscosity * (k**2)
    return {"kinetic_energy": k, "dissipation_rate": epsilon}

# Generate output file
def save_output_file(results, output_path="fluid_simulation_output.json"):
    """Stores computed fluid properties in JSON format."""
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

# Main function: Load input, process calculations, enforce stability, save output
def main(input_file_path, output_file_path, dx=0.01 * ureg.meter, dt=0.001 * ureg.second):
    """Executes the fluid dynamics calculation pipeline."""
    input_data = load_input_file(input_file_path)

    # Simulate grid spacing dynamically
    dx_field = np.full((100, 100), dx.magnitude, dtype=float)

    # Solve Navier-Stokes equations
    fluid_results = solve_navier_stokes(input_data)

    # Enforce numerical stability
    enforce_cfl_condition(np.array(fluid_results["velocity"], dtype=float), dx_field, dt)

    # Compute turbulence model effects
    turbulence_results = compute_turbulence_rans(np.array(fluid_results["velocity"], dtype=float), input_data["viscosity"].magnitude)

    # Combine results
    results = {
        "fluid_properties": fluid_results,
        "turbulence": turbulence_results,
        "cfl_condition": "Validated"
    }

    # Save results to output file
    save_output_file(results, output_file_path)

# Example execution
if __name__ == "__main__":
    input_file_path = "data/testing-input-output/boundary_conditions.json"
    output_file_path = "fluid_simulation_output.json"
    main(input_file_path, output_file_path)
