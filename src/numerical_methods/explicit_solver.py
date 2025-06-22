# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

# Import the individual numerical methods you already have
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    # UPDATED: Changed 'correct_pressure' to 'apply_pressure_correction'
    from .pressure_correction import apply_pressure_correction
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
    print("Please ensure advection.py, diffusion.py, pressure_divergence.py, "
          "poisson_solver.py, and pressure_correction.py exist in src/numerical_methods/ "
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
    dt: float
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

    Returns:
        tuple[np.ndarray, np.ndarray]: Updated velocity_field and pressure_field after one time step.

    Notes on assumed function signatures for imported modules:
    - compute_advection_term(field, velocity_field, mesh_info): Should return advection term.
    - compute_diffusion_term(field, viscosity, mesh_info): Should return diffusion term.
    - compute_pressure_divergence(u_tentative, mesh_info): Should return divergence field.
    - solve_poisson_for_phi(source_term, mesh_info, time_step, tolerance, max_iter): Should return solved scalar field (e.g., pressure correction).
    - apply_pressure_correction(u_tentative, current_pressure, phi, mesh_info, time_step, rho): Should return corrected velocity and updated pressure.
    """

    # Get grid shape from velocity_field for mesh_info
    grid_shape = velocity_field.shape[:-1] # (nx, ny, nz)

    # Prepare mesh_info dictionary for advection, diffusion, divergence, and poisson solver functions
    mesh_info = {
        'grid_shape': grid_shape,
        'dx': dx,
        'dy': dy,
        'dz': dz
    }

    # Start with a copy to avoid modifying the input field prematurely
    u_star = np.copy(velocity_field)

    # 1. Advection Step
    # Calculate the advection term for each velocity component
    # Advection term is - (u . grad)u. Apply as a change.
    advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, mesh_info)
    advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, mesh_info)
    advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, mesh_info)

    # Apply advection to u_star (minus sign because advection term is positive in N-S equation, but it's a "loss" from current perspective)
    u_star[..., 0] = velocity_field[..., 0] - dt * advection_u
    u_star[..., 1] = velocity_field[..., 1] - dt * advection_v
    u_star[..., 2] = velocity_field[..., 2] - dt * advection_w

    # 2. Diffusion Step
    # Calculate the diffusion term for each velocity component
    # Diffusion term is (viscosity * nabla^2(field)). Apply as a change.
    # Note: The term in N-S equation is (mu/rho) * nabla^2(u)
    diffusion_u = compute_diffusion_term(u_star[..., 0], viscosity, mesh_info)
    diffusion_v = compute_diffusion_term(u_star[..., 1], viscosity, mesh_info)
    diffusion_w = compute_diffusion_term(u_star[..., 2], viscosity, mesh_info)

    # Apply diffusion to u_star (positive sign for gain, divided by density to get acceleration)
    u_star[..., 0] += dt * (diffusion_u / density)
    u_star[..., 1] += dt * (diffusion_v / density)
    u_star[..., 2] += dt * (diffusion_w / density)


    # 3. Add External Forces (if any, e.g., gravity - not in current schema, but common)
    # u_star[..., 2] -= 9.81 * dt # Example for gravity in -z direction

    # --- Step 2: Pressure Projection (ensures incompressibility) ---
    # Compute the divergence of the intermediate velocity field
    divergence = compute_pressure_divergence(u_star, mesh_info)

    # Solve Poisson equation for pressure correction (phi)
    poisson_tolerance = 1e-6
    poisson_max_iter = 1000
    
    # The `solve_poisson_for_phi` function takes `divergence_u_star` directly, and divides by `time_step` internally.
    pressure_correction = solve_poisson_for_phi(
        divergence,                   # Corresponds to divergence_u_star
        mesh_info,                    # Contains dx, dy, dz and grid_shape
        dt,                           # Corresponds to time_step
        tolerance=poisson_tolerance,  # Pass tolerance
        max_iterations=poisson_max_iter # Pass max_iterations
    )

    # UPDATED: The apply_pressure_correction function now returns BOTH the updated velocity and pressure.
    # Removed the redundant pressure update line here.
    updated_velocity_field, updated_pressure_field = apply_pressure_correction(
        u_star,
        pressure_field,       # Pass the current pressure to be updated within the function
        pressure_correction,  # The calculated phi
        mesh_info,            # Contains dx, dy, dz, grid_shape
        dt,                   # Time step
        density               # Fluid density
    )

    return updated_velocity_field, updated_pressure_field

# Note: The actual implementation details within compute_advection_term, compute_diffusion_term, compute_pressure_divergence,
# solve_poisson_for_phi, and apply_pressure_correction will dictate the exact mathematical operations.
# You will need to ensure the arguments and return values match what is expected here.
