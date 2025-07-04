# src/numerical_methods/poisson_solver.py

import numpy as np
import sys

def _check_nan_inf(field, name, step_number, output_frequency_steps):
    """Helper to check for NaN/Inf and print debug info conditionally."""
    if step_number % output_frequency_steps == 0:
        has_nan = np.isnan(field).any()
        has_inf = np.isinf(field).any()
        min_val = np.min(field) if not has_nan and not has_inf else float('nan')
        max_val = np.max(field) if not has_nan and not has_inf else float('nan')
        print(f"  [Poisson Solver DEBUG] {name} stats BEFORE clamp: min={min_val:.2e}, max={max_val:.2e}, has_nan={has_nan}, has_inf={has_inf}")

    if np.isnan(field).any() or np.isinf(field).any():
        if step_number % output_frequency_steps == 0:
            print(f"  ❌ Warning: Invalid values in {name} — clamping to zero.")
        field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

    if step_number % output_frequency_steps == 0:
        print(f"  [Poisson Solver DEBUG] {name} stats AFTER clamp: min={np.min(field):.2e}, max={np.max(field):.2e}")
    return field


def solve_poisson_for_phi(
    velocity_star: np.ndarray,
    mesh_info: dict,
    time_step: float,
    density: float,
    pressure_field: np.ndarray, # Current pressure field for BC initialization
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    relaxation_factor: float = 1.2, # SOR relaxation factor
    step_number: int = 0, # Added for conditional logging
    output_frequency_steps: int = 1 # Added for conditional logging
) -> np.ndarray:
    """
    Solves the Poisson equation for the pressure correction potential (phi) using Successive Over-Relaxation (SOR).
    The equation is ∇²φ = (∇·u*) / Δt.

    Args:
        velocity_star (np.ndarray): Tentative velocity field (u*) with ghost cells.
                                    Shape: (nx+2, ny+2, nz+2, 3).
        mesh_info (dict): Dictionary containing grid spacing ('dx', 'dy', 'dz') and boundary conditions.
        time_step (float): Time step Δt.
        density (float): Fluid density ρ.
        pressure_field (np.ndarray): The current pressure field, used to apply pressure boundary conditions
                                     to the phi field.
        max_iterations (int): Maximum number of iterations for the SOR solver.
        tolerance (float): Convergence tolerance for the residual.
        relaxation_factor (float): Relaxation factor for SOR (ω).
        step_number (int): Current simulation step number, used for conditional logging.
        output_frequency_steps (int): Frequency for printing debug output, used for conditional logging.

    Returns:
        np.ndarray: The pressure correction potential (φ) field with ghost cells.
                    Shape: (nx+2, ny+2, nz+2).
    """
    dx, dy, dz = mesh_info["dx"], mesh_info["dy"], mesh_info["dz"]
    nx, ny, nz = velocity_star.shape[0] - 2, velocity_star.shape[1] - 2, velocity_star.shape[2] - 2

    # Initialize phi field with zeros, including ghost cells
    phi = np.zeros_like(pressure_field) # Start with current pressure field shape

    # Calculate the right-hand side (RHS) of the Poisson equation: RHS = (∇·u*) / Δt
    # Divergence ∇·u* = du*/dx + dv*/dy + dw*/dz
    # Note: velocity_star already includes ghost cells.
    # The gradient calculations below will operate on the interior cells.

    # du*/dx using central difference for interior cells
    du_dx = (velocity_star[2:, 1:-1, 1:-1, 0] - velocity_star[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
    # dv*/dy using central difference for interior cells
    dv_dy = (velocity_star[1:-1, 2:, 1:-1, 1] - velocity_star[1:-1, :-2, 1:-1, 1]) / (2 * dy)
    # dw*/dz using central difference for interior cells
    dw_dz = (velocity_star[1:-1, 1:-1, 2:, 2] - velocity_star[1:-1, 1:-1, :-2, 2]) / (2 * dz)

    divergence = du_dx + dv_dy + dw_dz

    # Clamp divergence to prevent NaN/Inf propagation
    if np.isnan(divergence).any() or np.isinf(divergence).any():
        if step_number % output_frequency_steps == 0:
            print("❌ Warning: Invalid values in divergence — clamping to zero.")
        divergence = np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0)

    # RHS of Poisson equation: (∇·u*) / Δt
    rhs = divergence / time_step

    # Apply boundary conditions to the phi field
    # For pressure Dirichlet BCs, phi is also Dirichlet (often phi=0 or a scaled value related to pressure)
    # For velocity Dirichlet BCs (e.g., walls), phi is Neumann (zero normal gradient)
    if step_number % output_frequency_steps == 0:
        print("[Poisson Solver] Initializing phi field and applying boundary conditions...")

    processed_bcs = mesh_info.get("boundary_conditions", {})
    for bc_name, bc in processed_bcs.items():
        if "cell_indices" not in bc or "ghost_indices" not in bc:
            if step_number % output_frequency_steps == 0:
                print(f"WARNING: BC '{bc_name}' is missing indices. Skipping. Was pre-processing successful?", file=sys.stderr)
            continue

        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
        cell_indices = np.array(bc["cell_indices"], dtype=int)
        ghost_indices = np.array(bc["ghost_indices"], dtype=int)

        if "pressure" in apply_to_fields and bc_type == "dirichlet":
            # For Dirichlet pressure, phi is often set to a scaled version of the pressure difference
            # or simply 0 if the pressure is absolute. Here, we'll use the pressure value itself
            # for consistency with the pressure update P_new = P_old + rho * phi.
            # If the inlet pressure is P_in and outlet is P_out, then phi could be (P_in - P_out) / rho.
            # For simplicity, let's assume phi should reflect the pressure at the boundary.
            # A common approach for pressure-correction is to set phi=0 at one pressure boundary
            # and use Neumann for others. Given your setup, we'll set phi directly based on pressure BC.
            if ghost_indices.size > 0:
                target_pressure = bc.get("pressure", 0.0)
                # Setting phi in ghost cells based on target pressure.
                # The exact scaling might depend on the specific formulation of the Poisson equation.
                # For now, let's try setting phi to the target pressure divided by density.
                # This makes phi consistent with the pressure update P_new = P_old + rho * phi.
                phi[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure / density
                if step_number % output_frequency_steps == 0:
                    print(f"   -> Applied Dirichlet phi ({target_pressure / density:.1f}) for pressure BC '{bc_name}'.")

        elif "velocity" in apply_to_fields and (bc_type == "dirichlet" or bc_type == "neumann" or bc_type == "outflow"):
            # For velocity boundaries (like walls, or outflow where velocity gradient is zero),
            # the normal derivative of phi is typically zero (Neumann boundary condition for phi).
            # This means ghost cell phi = adjacent interior cell phi.
            if cell_indices.size > 0 and ghost_indices.size > 0:
                phi[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = \
                    phi[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]]
                if step_number % output_frequency_steps == 0:
                    print(f"   -> Applied Neumann BC (zero normal gradient) to phi for velocity BC '{bc_name}'.")

    if step_number % output_frequency_steps == 0:
        print(f"[Poisson Solver] Starting SOR solver with {max_iterations} iterations and tolerance {tolerance:.1e}.")

    # SOR Iteration
    for it in range(max_iterations):
        old_phi = phi[1:-1, 1:-1, 1:-1].copy() # Copy interior cells for residual calculation

        # Iterate over interior cells
        for k in range(1, nz + 1):
            for j in range(1, ny + 1):
                for i in range(1, nx + 1):
                    # Compute Laplacian terms for phi at (i,j,k) using current phi values
                    # 1/dx^2 * (phi[i+1,j,k] - 2*phi[i,j,k] + phi[i-1,j,k])
                    # 1/dy^2 * (phi[i,j+1,k] - 2*phi[i,j,k] + phi[i,j-1,k])
                    # 1/dz^2 * (phi[i,j,k+1] - 2*phi[i,j,k] + phi[i,j,k-1])

                    # Note: The equation is ∇²φ = RHS.
                    # In finite difference, this becomes:
                    # (phi_i+1 - 2phi_i + phi_i-1)/dx^2 + (phi_j+1 - 2phi_j + phi_j-1)/dy^2 + (phi_k+1 - 2phi_k + phi_k-1)/dz^2 = RHS_ijk
                    # Rearranging for phi_i,j,k:
                    # phi_i,j,k = 1 / (2/dx^2 + 2/dy^2 + 2/dz^2) * [ RHS_ijk + (phi_i+1 + phi_i-1)/dx^2 + (phi_j+1 + phi_j-1)/dy^2 + (phi_k+1 + phi_k-1)/dz^2 ]

                    # Calculate the new phi value using the SOR formula
                    term_x = (phi[i+1, j, k] + phi[i-1, j, k]) / (dx**2)
                    term_y = (phi[i, j+1, k] + phi[i, j-1, k]) / (dy**2)
                    term_z = (phi[i, j, k+1] + phi[i, j, k-1]) / (dz**2)

                    denominator = 2 * (1/(dx**2) + 1/(dy**2) + 1/(dz**2))

                    phi_new_unrelaxed = (rhs[i-1, j-1, k-1] + term_x + term_y + term_z) / denominator

                    # Apply relaxation
                    phi[i, j, k] = (1 - relaxation_factor) * phi[i, j, k] + relaxation_factor * phi_new_unrelaxed

        # Calculate residual for convergence check
        # Residual = RHS - ∇²φ
        # ∇²φ_calculated = (phi[2:, 1:-1, 1:-1] - 2*phi[1:-1, 1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1]) / (dx**2) + \
        #                  (phi[1:-1, 2:, 1:-1] - 2*phi[1:-1, 1:-1, 1:-1] + phi[1:-1, :-2, 1:-1]) / (dy**2) + \
        #                  (phi[1:-1, 1:-1, 2:] - 2*phi[1:-1, 1:-1, 1:-1] + phi[1:-1, 1:-1, :-2]) / (dz**2)

        # A simpler way to calculate residual is based on the change in phi
        residual = np.max(np.abs(phi[1:-1, 1:-1, 1:-1] - old_phi))

        if residual < tolerance:
            if step_number % output_frequency_steps == 0:
                print(f"[Poisson Solver] Solver converged at iteration {it+1}. Final residual: {residual:.6e}")
            break
        elif it == max_iterations - 1:
            if step_number % output_frequency_steps == 0:
                print(f"WARNING: Poisson solver did NOT converge after {max_iterations} iterations. Final residual: {residual:.6e}", file=sys.stderr)

    # Apply boundary conditions to phi one last time after convergence,
    # especially important for Neumann boundaries.
    for bc_name, bc in processed_bcs.items():
        if "cell_indices" not in bc or "ghost_indices" not in bc:
            continue

        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
        cell_indices = np.array(bc["cell_indices"], dtype=int)
        ghost_indices = np.array(bc["ghost_indices"], dtype=int)

        if "pressure" in apply_to_fields and bc_type == "dirichlet":
            if ghost_indices.size > 0:
                target_pressure = bc.get("pressure", 0.0)
                phi[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = target_pressure / density

        elif "velocity" in apply_to_fields and (bc_type == "dirichlet" or bc_type == "neumann" or bc_type == "outflow"):
            if cell_indices.size > 0 and ghost_indices.size > 0:
                phi[ghost_indices[:, 0], ghost_indices[:, 1], ghost_indices[:, 2]] = \
                    phi[cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]]

    # Final check and clamp for phi
    phi = _check_nan_inf(phi, "phi output", step_number, output_frequency_steps)

    if step_number % output_frequency_steps == 0:
        print(f"[Poisson Solver] Solver finished. Final residual: {residual:.6e}") # Print final residual again for clarity

    return phi



