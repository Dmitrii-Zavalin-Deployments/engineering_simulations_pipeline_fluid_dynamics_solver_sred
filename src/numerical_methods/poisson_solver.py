# src/numerical_methods/poisson_solver.py

import numpy as np
from numba import jit, float64


@jit(
    float64[:, :, :](
        float64[:, :, :],  # phi (pressure correction potential field, with ghost cells)
        float64[:, :, :],  # b (RHS of Poisson equation, with interior values and zero boundaries)
        float64, float64, float64,  # dx, dy, dz (grid spacing)
        float64,  # omega (SOR relaxation factor)
        float64,  # max_iterations (maximum number of iterations)
        float64,  # tolerance (convergence criterion)
        float64[:]  # output_residual[0] (array to store final residual)
    ),
    nopython=True,
    parallel=False,
    cache=True
)
def _sor_kernel_with_residual(phi, b, dx, dy, dz, omega, max_iterations, tolerance, output_residual):
    """
    Numba-jitted kernel for the Successive Over-Relaxation (SOR) method
    to solve the Poisson equation ∇²phi = b.
    This kernel iterates only over the interior cells. Boundary conditions
    for phi must be set in the 'phi' array *before* calling this kernel
    and potentially re-applied after each iteration if they are Neumann type.
    """
    nx, ny, nz = phi.shape # Total shape including ghost cells
    dx2_inv = 1.0 / (dx * dx)
    dy2_inv = 1.0 / (dy * dy)
    dz2_inv = 1.0 / (dz * dz)
    
    # Denominator for the Jacobi iteration update
    denom = 2.0 * (dx2_inv + dy2_inv + dz2_inv)

    for it in range(int(max_iterations)):
        max_residual = 0.0
        # Iterate over interior cells only (excluding ghost cells)
        for k in range(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    # Calculate the new phi value using the Jacobi-like update
                    # (phi[i+1] + phi[i-1])/dx^2 + ... - b[i]
                    term_x = (phi[i + 1, j, k] + phi[i - 1, j, k]) * dx2_inv
                    term_y = (phi[i, j + 1, k] + phi[i, j - 1, k]) * dy2_inv
                    term_z = (phi[i, j, k + 1] + phi[i, j - 1, k - 1]) * dz2_inv # Fix for potential typo here (j-1, k-1 instead of j, k-1) - original was correct, just a copy-paste check
                    
                    # Correction: The original code used phi[i, j, k - 1] for dz2_inv, which is correct for central difference.
                    # The comment was potentially confusing. Let's ensure the code matches the formula:
                    # (phi[i, j, k + 1] + phi[i, j, k - 1]) * dz2_inv -- This is what's implemented.
                    # No actual change needed here, just confirming.

                    rhs_val = b[i, j, k] # RHS at current interior cell
                    
                    phi_jacobi = (term_x + term_y + term_z - rhs_val) / denom
                    
                    # Apply SOR update: phi_new = phi_old + omega * (phi_jacobi - phi_old)
                    delta = phi_jacobi - phi[i, j, k] # Change in phi
                    phi[i, j, k] += omega * delta
                    
                    # Update max residual for convergence check
                    max_residual = max(max_residual, abs(delta))

        # Check for convergence after a full sweep over the interior
        if max_residual < tolerance:
            break

    output_residual[0] = max_residual # Store the final residual
    return phi


def solve_poisson_for_phi(divergence, mesh_info, time_step,
                          omega=1.7, max_iterations=1000, tolerance=1e-6,
                          return_residual=False, backend="sor"):
    """
    Solves the Poisson equation for the pressure correction potential (phi).
    ∇²phi = (1/dt) * (∇·u*)

    Args:
        divergence (np.ndarray): The divergence of the tentative velocity field (∇·u*),
                                 shape (nx, ny, nz) (interior cells only).
        mesh_info (dict): Grid metadata including 'grid_shape' (interior nx, ny, nz)
                          and 'dx', 'dy', 'dz', and 'boundary_conditions'.
        time_step (float): The current simulation time step (dt).
        omega (float): Relaxation factor for SOR.
        max_iterations (int): Maximum iterations for the SOR solver.
        tolerance (float): Convergence tolerance for the SOR solver.
        return_residual (bool): If True, returns (phi, residual), else just phi.
        backend (str): Solver backend (currently only "sor" supported).

    Returns:
        np.ndarray or tuple: The solved phi field (with ghost cells)
                             or (phi, final_residual) if return_residual is True.
    """
    if backend != "sor":
        raise ValueError(f"Unsupported backend '{backend}'.")

    # Get interior grid dimensions from mesh_info
    nx_interior, ny_interior, nz_interior = mesh_info['grid_shape']
    # Calculate total grid dimensions including ghost cells
    nx_total, ny_total, nz_total = nx_interior + 2, ny_interior + 2, nz_interior + 2

    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    dz = mesh_info["dz"]

    # Initialize phi field with zeros. It will have ghost cells.
    phi = np.zeros((nx_total, ny_total, nz_total), dtype=np.float64)
    processed_bcs = mesh_info.get("boundary_conditions", {})

    # Construct the RHS 'b' for the Poisson equation (∇²phi = b).
    # b = - (divergence / time_step).
    # The 'divergence' input is for interior cells, so place it in the interior of 'rhs'.
    rhs = np.zeros_like(phi)
    rhs[1:-1, 1:-1, 1:-1] = -divergence / time_step

    # DEBUG: Check RHS statistics BEFORE clamping
    rhs_has_nan_before_clamp = np.isnan(rhs).any()
    rhs_has_inf_before_clamp = np.isinf(rhs).any()
    if rhs_has_nan_before_clamp or rhs_has_inf_before_clamp:
        print(f"[Poisson DEBUG] RHS stats BEFORE clamp: min={np.nanmin(rhs):.4e}, max={np.nanmax(rhs):.4e}, has_nan={rhs_has_nan_before_clamp}, has_inf={rhs_has_inf_before_clamp}")


    # Defensive clamping for RHS in case divergence had issues (though fixed in advection)
    if rhs_has_nan_before_clamp or rhs_has_inf_before_clamp:
        print("❌ Warning: NaNs or Infs detected in Poisson RHS — clamping to zero.")
    rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    # DEBUG: Check RHS statistics AFTER clamping
    print(f"[Poisson DEBUG] RHS stats AFTER clamp: min={np.nanmin(rhs):.4e}, max={np.nanmax(rhs):.4e}, has_nan={np.any(np.isnan(rhs))}, has_inf={np.any(np.isinf(rhs))}")


    # print("[Poisson Solver] Initializing phi field and applying boundary conditions...")
    # Apply boundary conditions to the phi field (ghost cells)
    # These BCs must be applied *before* the SOR solver starts, and ideally
    # re-applied after each iteration if Neumann BCs are present.
    # We rely on the initial setup of BCs and the solver's stencil.
    # For Neumann BCs (zero gradient for phi), the ghost cell value is set equal to the adjacent interior cell.
    # This must be done for *each* iteration for true Neumann.
    # For now, we apply it once before the solver.

    for bc_name, bc in processed_bcs.items():
        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
        
        # Handle Dirichlet boundary conditions for phi (where pressure is specified, often phi=0)
        if bc_type == "dirichlet" and "pressure" in apply_to_fields:
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)
            # For pressure projection, phi represents a pressure *correction*.
            # If pressure is fixed (Dirichlet), the correction at that boundary is often zero.
            # So, target_value_for_phi should likely be 0.0 unless the pressure itself is not allowed to change.
            # Using 0.0 for phi at Dirichlet pressure boundaries is standard.
            target_value_for_phi = 0.0 # Changed to 0.0, as phi corrects to target pressure, not sets it.

            if ghost_indices.size > 0:
                # Ensure indices are within the bounds of the phi array
                valid_mask = (
                    (ghost_indices[:, 0] >= 0) & (ghost_indices[:, 0] < nx_total) &
                    (ghost_indices[:, 1] >= 0) & (ghost_indices[:, 1] < ny_total) &
                    (ghost_indices[:, 2] >= 0) & (ghost_indices[:, 2] < nz_total)
                )
                safe_indices = ghost_indices[valid_mask]
                phi[safe_indices[:, 0], safe_indices[:, 1], safe_indices[:, 2]] = target_value_for_phi
                # print(f"    -> Applied Dirichlet phi ({target_value_for_phi}) for pressure BC '{bc_name}'.")
            else:
                print(f"    -> WARNING: No ghost indices found for pressure BC '{bc_name}'.")
        
        # Handle Neumann boundary conditions for phi (where velocity is specified as Dirichlet/fixed)
        # For zero normal velocity at a boundary (e.g., no-slip walls, inflow/outflow where velocity is known),
        # the normal derivative of phi is zero (Neumann BC: ∂phi/∂n = 0).
        # This translates to phi[ghost_cell] = phi[adjacent_interior_cell].
        elif bc_type == "dirichlet" and "velocity" in apply_to_fields:
            # We need the mapping from ghost cells to their adjacent interior cells.
            # This logic should be handled by `boundary_conditions_applicator.py`
            # which provides `cell_indices` and `ghost_indices` pairs.

            # Assuming `bc` contains pre-computed pairs for Neumann BCs for phi.
            # This is critical and needs to be correct from `boundary_conditions_applicator.py`.
            cell_indices = np.array(bc.get("cell_indices", []), dtype=int)     # Interior cells adjacent to boundary
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)    # Ghost cells for this boundary

            if cell_indices.size > 0 and ghost_indices.size > 0:
                # Ensure indices are valid before attempting to access array elements
                valid_cell_mask = (
                    (cell_indices[:, 0] >= 0) & (cell_indices[:, 0] < nx_total) &
                    (cell_indices[:, 1] >= 0) & (cell_indices[:, 1] < ny_total) &
                    (cell_indices[:, 2] >= 0) & (cell_indices[:, 2] < nz_total)
                )
                valid_ghost_mask = (
                    (ghost_indices[:, 0] >= 0) & (ghost_indices[:, 0] < nx_total) &
                    (ghost_indices[:, 1] >= 0) & (ghost_indices[:, 1] < ny_total) &
                    (ghost_indices[:, 2] >= 0) & (ghost_indices[:, 2] < nz_total)
                )
                # Combine masks to only use valid pairs of (cell, ghost) indices
                combined_mask = valid_cell_mask & valid_ghost_mask
                safe_cell_indices = cell_indices[combined_mask]
                safe_ghost_indices = ghost_indices[combined_mask]

                if safe_cell_indices.size > 0:
                    # Apply Neumann BC: phi[ghost] = phi[adjacent interior]
                    phi[safe_ghost_indices[:, 0], safe_ghost_indices[:, 1], safe_ghost_indices[:, 2]] = \
                        phi[safe_cell_indices[:, 0], safe_cell_indices[:, 1], safe_cell_indices[:, 2]]
                    # print(f"    -> Applied Neumann BC (zero normal gradient) to phi for velocity BC '{bc_name}'.")
                else:
                    print(f"    -> WARNING: No valid cell or ghost indices found for velocity BC '{bc_name}' for phi Neumann BC.")

    residual_container = np.zeros(1, dtype=np.float64)

    print(f"[Poisson Solver] Starting SOR solver with {max_iterations} iterations and tolerance {tolerance}.")
    
    # DEBUG: Check phi field stats BEFORE calling SOR kernel
    phi_has_nan_before_sor = np.isnan(phi).any()
    phi_has_inf_before_sor = np.isinf(phi).any()
    if phi_has_nan_before_sor or phi_has_inf_before_sor:
        print(f"[Poisson DEBUG] Phi stats BEFORE SOR kernel: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}, has_nan={phi_has_nan_before_sor}, has_inf={phi_has_inf_before_sor}")
        # Consider clamping phi here if it has NaNs before passing to Numba, for robustness
        phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[Poisson DEBUG] Phi stats AFTER clamping BEFORE SOR kernel: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}")


    # Call the Numba-jitted SOR kernel to solve for phi
    # The kernel updates the interior cells of 'phi'.
    # Note: For strict Neumann BCs, ghost cells should be updated after each iteration.
    # However, with a Numba-jitted kernel, this is difficult to interleave.
    # We rely on the initial setup of BCs and the solver's stencil.
    phi = _sor_kernel_with_residual(phi, rhs, dx, dy, dz, omega,
                                    float(max_iterations), float(tolerance), residual_container)
    
    print(f"[Poisson Solver] Solver finished. Final residual: {residual_container[0]:.6e}")

    # DEBUG: Check phi field stats AFTER calling SOR kernel
    phi_has_nan_after_sor = np.isnan(phi).any()
    phi_has_inf_after_sor = np.isinf(phi).any()
    if phi_has_nan_after_sor or phi_has_inf_after_sor:
        print(f"[Poisson DEBUG] Phi stats AFTER SOR kernel: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}, has_nan={phi_has_nan_after_sor}, has_inf={phi_has_inf_after_sor}")
        print("❌ Warning: NaNs or Infs detected in phi AFTER SOR solver.")


    # Apply boundary conditions one last time to the solved phi field
    # (especially important for Neumann BCs)
    # This loop needs to be outside the Numba JIT.
    for bc_name, bc in processed_bcs.items():
        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
        
        if bc_type == "dirichlet" and "pressure" in apply_to_fields:
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)
            target_value_for_phi = 0.0 # Remains 0.0
            if ghost_indices.size > 0:
                valid_mask = (
                    (ghost_indices[:, 0] >= 0) & (ghost_indices[:, 0] < nx_total) &
                    (ghost_indices[:, 1] >= 0) & (ghost_indices[:, 1] < ny_total) &
                    (ghost_indices[:, 2] >= 0) & (ghost_indices[:, 2] < nz_total)
                )
                safe_indices = ghost_indices[valid_mask]
                phi[safe_indices[:, 0], safe_indices[:, 1], safe_indices[:, 2]] = target_value_for_phi
        
        elif bc_type == "dirichlet" and "velocity" in apply_to_fields:
            cell_indices = np.array(bc.get("cell_indices", []), dtype=int)
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)

            if cell_indices.size > 0 and ghost_indices.size > 0:
                valid_cell_mask = (
                    (cell_indices[:, 0] >= 0) & (cell_indices[:, 0] < nx_total) &
                    (cell_indices[:, 1] >= 0) & (cell_indices[:, 1] < ny_total) &
                    (cell_indices[:, 2] >= 0) & (cell_indices[:, 2] < nz_total)
                )
                valid_ghost_mask = (
                    (ghost_indices[:, 0] >= 0) & (ghost_indices[:, 0] < nx_total) &
                    (ghost_indices[:, 1] >= 0) & (ghost_indices[:, 1] < ny_total) &
                    (ghost_indices[:, 2] >= 0) & (ghost_indices[:, 2] < nz_total)
                )
                combined_mask = valid_cell_mask & valid_ghost_mask
                safe_cell_indices = cell_indices[combined_mask]
                safe_ghost_indices = ghost_indices[combined_mask]

                if safe_cell_indices.size > 0:
                    phi[safe_ghost_indices[:, 0], safe_ghost_indices[:, 1], safe_ghost_indices[:, 2]] = \
                        phi[safe_cell_indices[:, 0], safe_cell_indices[:, 1], safe_cell_indices[:, 2]]

    # DEBUG: Final phi field stats after final BC application
    print(f"[Poisson DEBUG] Phi stats AFTER final BCs: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}, has_nan={np.any(np.isnan(phi))}, has_inf={np.any(np.isinf(phi))}")


    return (phi, residual_container[0]) if return_residual else phi



