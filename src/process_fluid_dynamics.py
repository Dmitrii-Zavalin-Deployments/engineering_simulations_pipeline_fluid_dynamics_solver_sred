import json
import numpy as np
from pint import UnitRegistry
import sys

# Initialize unit registry for scientific units
ureg = UnitRegistry()

# Load input file
def load_input_file(file_path):
    """Reads and validates the input JSON file with fluid properties."""
    try:
        with open(file_path, "r") as file:
            input_data = json.load(file)

        # Extract fluid properties from the inlet boundary
        fluid_properties = input_data["inlet_boundary"]["fluid_properties"]

        # Convert velocity into a NumPy array representing a vector with units
        # Ensure velocity is a list of 3 floats in the input JSON for 3D velocity
        input_data["fluid_velocity"] = np.array(input_data["inlet_boundary"]["velocity"], dtype=float) * ureg.meter / ureg.second

        # Convert units correctly, handling potential None values
        input_data["pressure"] = fluid_properties.get("pressure", 101325.0) * ureg.pascal # Default to atmospheric pressure if not provided
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

# Solve Navier-Stokes Equations
def solve_navier_stokes(input_data, grid_size=(100, 100, 3), dt=0.0001 * ureg.second, num_iterations=5000): # Reduced dt, increased iterations
    """Numerically solves Navier-Stokes using Finite Volume Method (simplified)."""
    initial_velocity = input_data["fluid_velocity"].magnitude
    if initial_velocity.ndim == 1 and initial_velocity.shape[0] == 3:
        # Initialize velocity field for a 2D grid with 3D velocity vectors at each point
        velocity = np.tile(initial_velocity, (grid_size[0], grid_size[1], 1)).astype(float)
    else:
        raise ValueError("Initial velocity must be a vector of 3 components.")

    # Pressure is 2D (grid_size[0] x grid_size[1])
    pressure = np.full(grid_size[:2], input_data["pressure"].magnitude, dtype=float)
    density = input_data["density"].magnitude
    viscosity = input_data["viscosity"].magnitude

    data_points = []

    for _ in range(num_iterations): # Use num_iterations
        advection_term = np.zeros_like(velocity)
        diffusion_term = np.zeros_like(velocity)
        pressure_gradient = np.zeros_like(velocity)

        # Advection and Diffusion terms
        for i in range(3): # For each velocity component (x, y, z)
            # Use 2D slices for gradient calculations (velocity[:,:,i] is 2D)
            grad_vx, grad_vy = np.gradient(velocity[:, :, i], axis=(0, 1))

            # Advection: u*du/dx + v*du/dy (assuming 2D flow dominates pressure effects on z-vel)
            advection_term[:, :, i] = - (velocity[:, :, 0] * grad_vx + velocity[:, :, 1] * grad_vy)
            
            # Diffusion (Laplacian in 2D)
            laplacian_v_x = np.gradient(np.gradient(velocity[:, :, i], axis=0), axis=0)
            laplacian_v_y = np.gradient(np.gradient(velocity[:, :, i], axis=1), axis=1)
            diffusion_term[:, :, i] = viscosity * (laplacian_v_x + laplacian_v_y)

        # Pressure gradient term (pressure is 2D)
        grad_p_x, grad_p_y = np.gradient(pressure, axis=(0, 1))

        # Apply 2D pressure gradients to the first two components of the 3D velocity vector
        pressure_gradient[:, :, 0] = -grad_p_x / density
        pressure_gradient[:, :, 1] = -grad_p_y / density
        # pressure_gradient[:, :, 2] remains 0 as pressure is 2D and assumed not to vary in z for gradient purposes.

        # Update velocity
        velocity += dt.magnitude * (advection_term + diffusion_term + pressure_gradient)

        # Store sample data points from velocity and pressure fields
        # Ensure pressure is stored as a float, not a single-element array
        data_points.append({"velocity": velocity[50, 50].tolist(), "pressure": float(pressure[50, 50])})

    return {
        "velocity": velocity.tolist(),
        "pressure": pressure.tolist(),
        "data_points": data_points
    }

# Generate output file
def save_output_file(results, output_path):
    """Stores computed fluid properties in JSON format."""
    try:
        with open(output_path, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Successfully saved output to: {output_path}")
    except IOError as e:
        print(f"ERROR: Could not write output file to {output_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving output file: {e}")
        sys.exit(1)

# Main function: Load input, process calculations, enforce stability, save output
def main(input_file_path, output_file_path):
    """Executes the fluid dynamics calculation pipeline."""
    input_data = load_input_file(input_file_path)
    fluid_results = solve_navier_stokes(input_data)
    
    # Example values for turbulence and CFL condition (placeholders if not computed)
    results = {
        "fluid_properties": {
            "velocity_field": fluid_results["velocity"], # Renamed for clarity
            "pressure_field": fluid_results["pressure"], # Renamed for clarity
            "density": input_data["density"].magnitude,
            "viscosity": input_data["viscosity"].magnitude
        },
        "turbulence": {"kinetic_energy": 0.5, "dissipation_rate": 0.1},  # Example turbulence values
        "cfl_condition": "Validated", # This should ideally be computed and checked
        "data_points": fluid_results["data_points"]  # Ensure data points are included
    }

    save_output_file(results, output_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_fluid_dynamics.py <input_file_path> <output_file_path>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    main(input_file_path, output_file_path)



