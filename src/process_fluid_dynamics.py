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

    # Convert velocity into a NumPy array representing a vector with units
    input_data["fluid_velocity"] = np.array(input_data["inlet_boundary"]["velocity"], dtype=float) * ureg.meter / ureg.second

    # Convert units correctly, handling potential None values
    input_data["pressure"] = fluid_properties.get("pressure", 0) * ureg.pascal
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
    cfl_value = (max_velocity * dt.magnitude / min_dx) # Extract dt magnitude for computation
    assert cfl_value <= 1, f"CFL condition violated! CFL = {cfl_value:.4f}. Adjust time-step or grid spacing."

# Solve Navier-Stokes Equations
def solve_navier_stokes(input_data, grid_size=(100, 100, 3), dt=0.001 * ureg.second):
    """Numerically solves Navier-Stokes using Finite Volume Method."""
    initial_velocity = input_data["fluid_velocity"].magnitude # Get the magnitude
    if initial_velocity.ndim == 1 and initial_velocity.shape[0] == 3:
        velocity = np.tile(initial_velocity, (grid_size[0], grid_size[1], 1)) # Tile the vector
    else:
        raise ValueError("Initial velocity must be a vector of 3 components.")

    pressure = np.full(grid_size[:2], input_data["pressure"].magnitude, dtype=float) # Use magnitude for initialization
    density = input_data["density"].magnitude
    viscosity = input_data["viscosity"].magnitude  # Ensure viscosity is a float

    # Compute advection, diffusion, and pressure gradient separately for vx, vy, vz
    for _ in range(500):
        advection_term = np.zeros_like(velocity)
        diffusion_term = np.zeros_like(velocity)
        pressure_gradient = np.zeros_like(velocity)

        # Compute advection and diffusion separately per velocity component
        for i in range(3):
            grad_vx, grad_vy = np.gradient(velocity[..., i], axis=(0, 1))  # Compute gradients separately
            advection_term[..., i] = -velocity[..., i] * (grad_vx + grad_vy)  # Fix broadcasting issue
            # Ensure viscosity used in the calculation is the float magnitude
            grad_x_diff = np.gradient(velocity[..., i], axis=0)
            grad_y_diff = np.gradient(velocity[..., i], axis=1)
            laplacian = np.gradient(grad_x_diff, axis=0) + np.gradient(grad_y_diff, axis=1)
            diffusion_term[..., i] = viscosity * laplacian

        # Compute pressure gradient per velocity component
        grad_x, grad_y = np.gradient(pressure, axis=(0, 1))
        pressure_gradient[..., 0] = -grad_x / density  # Assign x component
        pressure_gradient[..., 1] = -grad_y / density  # Assign y component

        velocity += dt.magnitude * (advection_term + diffusion_term + pressure_gradient) # Extract dt magnitude for computation

    return {"velocity": velocity.tolist(), "pressure": pressure.tolist()}

# Compute Turbulence using RANS k-epsilon Model
def compute_turbulence_rans(velocity_field, viscosity):
    """Computes turbulence dissipation rate using k-epsilon model."""
    velocity_magnitude = np.linalg.norm(velocity_field, axis=-1)
    k = 0.5 * np.mean(velocity_magnitude**2)
    epsilon = viscosity.magnitude * (k**2)  # Use viscosity magnitude
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
    fluid_results = solve_navier_stokes(input_data, dt=dt) # Pass dt with units to the solver

    # Enforce numerical stability
    enforce_cfl_condition(np.array(fluid_results["velocity"], dtype=float), dx_field, dt)

    # Compute turbulence model effects
    turbulence_results = compute_turbulence_rans(np.array(fluid_results["velocity"], dtype=float), input_data["viscosity"])

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



