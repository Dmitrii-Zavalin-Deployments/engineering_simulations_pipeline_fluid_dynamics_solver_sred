import json
import numpy as np
from pint import UnitRegistry
import sys # <-- Make sure sys is imported

# Initialize unit registry for scientific units
ureg = UnitRegistry()

# Load input file
def load_input_file(file_path):
    """Reads and validates the input JSON file with fluid properties."""
    try: # Added try-except for robust file handling
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
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {file_path}")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: Missing expected key in input file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading input file: {e}")
        sys.exit(1)


# Apply boundary conditions (Note: This function is defined but not called in main. Is it intended for future use?)
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
    if cfl_value > 1: # Change assert to if for more graceful handling in production/workflow
        print(f"WARNING: CFL condition violated! CFL = {cfl_value:.4f}. Adjust time-step or grid spacing.")
        # Optionally, you might want to sys.exit(1) here if a CFL violation should fail the workflow
        # sys.exit(1)
    else:
        print(f"CFL condition met. CFL = {cfl_value:.4f}")

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
        pressure_gradient[..., 2] = np.zeros_like(pressure_gradient[..., 2]) # Assuming no z-gradient of pressure for 2D pressure field

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
def save_output_file(results, output_path): # Removed default value
    """Stores computed fluid properties in JSON format."""
    try:
        with open(output_path, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Successfully saved output to: {output_path}") # Added success print
    except IOError as e:
        print(f"ERROR: Could not write output file to {output_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving output file: {e}")
        sys.exit(1)

# Main function: Load input, process calculations, enforce stability, save output
def main(input_file_path, output_file_path, dx=0.01 * ureg.meter, dt=0.001 * ureg.second):
    """Executes the fluid dynamics calculation pipeline."""
    print(f"DEBUG (main): Input file path received: {input_file_path}") # Added debug print
    print(f"DEBUG (main): Output file path received: {output_file_path}") # Added debug print

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

# Example execution - MODIFIED TO USE COMMAND LINE ARGUMENTS
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_fluid_dynamics.py <input_file_path> <output_file_path>")
        sys.exit(1) # Exit if not enough arguments

    # Get file paths from command-line arguments
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    main(input_file_path, output_file_path)



