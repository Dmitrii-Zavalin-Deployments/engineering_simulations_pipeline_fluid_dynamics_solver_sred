# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

# Import the individual numerical methods you already have
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    from .pressure_correction import apply_pressure_correction
    # NEW IMPORT: This is crucial for applying boundary conditions
    from src.physics.boundary_conditions import apply_boundary_conditions
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
    print("Please ensure advection.py, diffusion.py, pressure_divergence.py, "
          "poisson_solver.py, pressure_correction.py, and boundary_conditions.py exist in their respective directories "
          "and contain the expected functions.", file=sys.stderr)
    sys.exit(1)


def solve_explicit(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    density: float,
    viscosity: float,
    dx: float,
    dy: float,
    dz: float,
    dt: float,
    mesh_info: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs one explicit time step for a fluid simulation using the Navier-Stokes equations.
    This function orchestrates advection, diffusion, and pressure projection steps.

    Args:
        velocity_field (np.ndarray): Current velocity field (U, V, W components)
                                     Shape: (nx, ny, nz, 3).
        pressure_field (np.ndarray): Current pressure field. Shape: (nx, ny, nz).
        density (float): Fluid density.
        viscosity (float): Fluid dynamic viscosity.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        dz (float): Grid spacing in z-direction.
        dt (float): Time step size.
        mesh_info (dict): Dictionary containing structured grid information and
                          pre-processed boundary condition data.

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated velocity_field and pressure_field after one time step.

    Notes on assumed function signatures for imported modules:
    - compute_advection_term(field, velocity_field, mesh_info): Should return advection term.
    - compute_diffusion_term(field, viscosity, mesh_info): Should return diffusion term.
    - compute_pressure_divergence(u_tentative, mesh_info): Should return divergence field.
    - solve_poisson_for_phi(source_term, mesh_info, time_step, tolerance, max_iter): Should return solved scalar field (e.g., pressure correction).
    - apply_pressure_correction(u_tentative, current_pressure, phi, mesh_info, time_step, rho): Should return corrected velocity and updated pressure.
    - apply_boundary_conditions(velocity_field, pressure_field, fluid_properties_dict, mesh_info, is_tentative_step): Should apply BCs to fields.
    """

    # Create fluid_properties_dict to pass to boundary conditions
    fluid_properties_dict = {
        "density": density,
        "viscosity": viscosity
    }

    # Start with a copy to avoid modifying the input field prematurely
    u_star = np.copy(velocity_field)

    # 1. Advection Step
    advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, mesh_info)
    advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, mesh_info)
    advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, mesh_info)

    u_star[..., 0] = velocity_field[..., 0] - dt * advection_u
    u_star[..., 1] = velocity_field[..., 1] - dt * advection_v
    u_star[..., 2] = velocity_field[..., 2] - dt * advection_w

    # 2. Diffusion Step
    diffusion_u = compute_diffusion_term(u_star[..., 0], viscosity, mesh_info)
    diffusion_v = compute_diffusion_term(u_star[..., 1], viscosity, mesh_info)
    diffusion_w = compute_diffusion_term(u_star[..., 2], viscosity, mesh_info)

    u_star[..., 0] += dt * (diffusion_u / density)
    u_star[..., 1] += dt * (diffusion_v / density)
    u_star[..., 2] += dt * (diffusion_w / density)

    # 3. Add External Forces (if any)
    # u_star[..., 2] -= 9.81 * dt # Example for gravity in -z direction

    # --- Apply Boundary Conditions to the intermediate velocity field (u_star) ---
    # u_star is a "tentative" velocity field, so is_tentative_step should be True.
    u_star, _ = apply_boundary_conditions(
        u_star,
        pressure_field,
        fluid_properties_dict, # Pass the new argument
        mesh_info,
        is_tentative_step=True # Crucial fix: indicate this is a tentative step
    )

    # --- Step 2: Pressure Projection (ensures incompressibility) ---
    # Compute the divergence of the intermediate velocity field
    divergence = compute_pressure_divergence(u_star, mesh_info)

    # Solve Poisson equation for pressure correction (phi)
    poisson_tolerance = 1e-6
    poisson_max_iter = 1000

    pressure_correction = solve_poisson_for_phi(
        divergence,
        mesh_info,
        dt,
        tolerance=poisson_tolerance,
        max_iterations=poisson_max_iter
    )

    # Apply pressure correction
    updated_velocity_field, updated_pressure_field = apply_pressure_correction(
        u_star,
        pressure_field,
        pressure_correction,
        mesh_info,
        dt,
        density
    )

    # --- Apply Boundary Conditions to the final updated fields after pressure correction ---
    # This ensures the final fields conform to boundary conditions.
    # These are the "final" fields for this time step, so is_tentative_step should be False.
    updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
        updated_velocity_field,
        updated_pressure_field,
        fluid_properties_dict, # Pass the new argument
        mesh_info,
        is_tentative_step=False # Crucial fix: indicate this is a final step
    )

    return updated_velocity_field, updated_pressure_field
