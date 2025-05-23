import sys
import os

# Add `src/` directory to Python module search path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from mesh import Mesh
from boundary_conditions import read_boundary_conditions, apply_boundary_conditions
from initialization import initialize_simulation
from momentum_solver import solve_momentum_equation
from pressure_solver import solve_pressure_correction
from velocity_correction import correct_velocity
from turbulence_solver import solve_turbulence_equations
from time_stepping import adjust_time_step
from convergence_check import check_convergence
from post_processing import export_results

# File paths
INPUT_JSON = os.path.join(os.path.dirname(__file__), "../data/testing-input-output/boundary_conditions.json")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "../data/testing-input-output/fluid_dynamics_results.json")

# Step 1: Read boundary conditions
boundary_data = read_boundary_conditions(INPUT_JSON)

# Step 2: Load mesh
mesh = Mesh(filename=os.path.join(os.path.dirname(__file__), "../data/mesh_file.msh"))

# Step 3: Apply boundary conditions
apply_boundary_conditions(mesh, boundary_data)

# Step 4: Initialize simulation variables
sim_settings = initialize_simulation(mesh, boundary_data)

# Extract simulation parameters
max_iterations = sim_settings["max_iterations"]
residual_tolerance = sim_settings["residual_tolerance"]
dt = sim_settings["suggested_time_step"]
CFL = sim_settings["CFL_condition"]

# Initialize velocity, pressure, and turbulence fields
U = [cell.velocity for cell in mesh.cells]
P = [cell.pressure for cell in mesh.cells]
k = [cell.k for cell in mesh.cells]
epsilon = [cell.epsilon for cell in mesh.cells]

# Step 5: Start CFD time loop
iteration = 0
while iteration < max_iterations:
    print(f"Iteration {iteration}")

    # Step 6: Solve momentum equation
    U_star = solve_momentum_equation(U, P, mesh)

    # Step 7: Solve pressure correction using multi-grid
    P_prime = solve_pressure_correction(P, U_star, mesh)

    # Step 8: Correct velocity using pressure gradient
    correct_velocity(U_star, P_prime, mesh)

    # Step 9: Solve turbulence transport equations (k-epsilon model)
    k, epsilon = solve_turbulence_equations(k, epsilon, U_star, mesh)

    # Step 10: Adjust time-step based on CFL condition
    dt = adjust_time_step(sim_settings, U_star, mesh)

    # Step 11: Check for convergence
    if check_convergence(U_star, P, k, epsilon, residual_tolerance):
        print("Convergence reached. Exiting loop.")
        break

    iteration += 1

# Step 12: Export results to JSON
export_results(mesh, U, P, k, epsilon, sim_settings, OUTPUT_JSON)

print("Simulation complete. Results saved to:", OUTPUT_JSON)



