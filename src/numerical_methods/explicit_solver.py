# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

# Import the individual numerical methods you already have
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    from .pressure_correction import apply_pressure_correction # Already updated in your provided code
    # NEW IMPORT: This is crucial for applying boundary conditions
    from src.physics.boundary_conditions import apply_boundary_conditions # Assuming this function exists or will be created
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
    mesh_info: dict # <<< ADDED THIS ARGUMENT HERE
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
    - apply_boundary_conditions(velocity_field, pressure_field, mesh_info): Should apply BCs to fields.
    """

    # Get grid shape from velocity_field for mesh_info
    # NOTE: The mesh_info dictionary is now PASSED IN, not created here.
    # The mesh_info should come fully prepared from pre_process_input.py
    # So, we can remove the manual mesh_info creation here.
    # However, your existing code uses 'grid_shape', 'dx', 'dy', 'dz'
    # which implies mesh_info would typically contain these.
    # We'll assume the passed mesh_info contains at least 'dx', 'dy', 'dz', 'grid_shape'
    # if these are still needed by other numerical methods called below.

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

    # --- NEW: Apply Boundary Conditions to the intermediate velocity field (u_star) ---
    # It's common to apply BCs after the advection and diffusion steps,
    # before the pressure projection, and again after pressure correction.
    # The `apply_boundary_conditions` function will use the `boundary_conditions_data`
    # stored within `mesh_info`.
    u_star, _ = apply_boundary_conditions(u_star, pressure_field, mesh_info) # Pressure field might not be modified here, depending on BC types.

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

    # --- NEW: Apply Boundary Conditions to the final updated fields after pressure correction ---
    # This ensures the final fields conform to boundary conditions.
    updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
        updated_velocity_field, updated_pressure_field, mesh_info
    )

    return updated_velocity_field, updated_pressure_field
