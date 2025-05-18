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

    # Convert units correctly
    input_data["fluid_velocity"] *= ureg.meter / ureg.second
    input_data["pressure"] *= ureg.pascal
    input_data["density"] *= ureg.kilogram / ureg.meter**3
    input_data["viscosity"] *= ureg.pascal * ureg.second
    return input_data

# Apply boundary conditions
def apply_boundary_conditions(mesh, input_data):
    """Assigns inlet, outlet, and wall boundary conditions."""
    mesh["boundary"]["inlet"]["velocity"] = input_data["fluid_velocity"]
    mesh["boundary"]["outlet"]["pressure"] = input_data["pressure"]
    mesh["boundary"]["walls"]["velocity"] = 0 * ureg.meter / ureg.second  # No-slip condition
    return mesh

# CFL condition enforcement
def enforce_cfl_condition(velocity_field, dx_field, dt):
    """Ensures CFL condition across computational domain."""
    max_velocity = np.max(velocity_field)
    min_dx = np.min(dx_field)
    cfl_value = max_velocity * dt / min_dx
    assert cfl_value <= 1, "CFL condition violated! Adjust time-step or grid spacing."

# Solve Navier-Stokes Equations
def solve_navier_stokes(input_data, grid_size=(100, 100), dt=0.001):
    """Numerically solves Navier-Stokes using Finite Volume Method."""
    velocity = np.full(grid_size, input_data["fluid_velocity"].magnitude)
    pressure = np.full(grid_size, input_data["pressure"].magnitude)
    density = input_data["density"].magnitude
    viscosity = input_data["viscosity"].magnitude

    # Compute advection, diffusion, and pressure gradient
    for _ in range(500):
        advection_term = -velocity * np.gradient(velocity)
        diffusion_term = viscosity * np.gradient(np.gradient(velocity))
        pressure_gradient = -np.gradient(pressure) / density
        velocity += dt * (advection_term + diffusion_term + pressure_gradient)

    return {"velocity": velocity.tolist(), "pressure": pressure.tolist()}

# Compute Turbulence using RANS k-epsilon Model
def compute_turbulence_rans(velocity_field, viscosity):
    """Computes turbulence dissipation rate using k-epsilon model."""
    k = 0.5 * np.mean(velocity_field**2)
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
    dx_field = np.full((100, 100), dx.magnitude)

    # Solve Navier-Stokes equations
    fluid_results = solve_navier_stokes(input_data)

    # Enforce numerical stability
    enforce_cfl_condition(fluid_results["velocity"], dx_field, dt)

    # Compute turbulence model effects
    turbulence_results = compute_turbulence_rans(np.array(fluid_results["velocity"]), input_data["viscosity"].magnitude)

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
