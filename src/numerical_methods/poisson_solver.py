# src/numerical_methods/poisson_solver.py

import numpy as np
from numba import jit, float64
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
import time # For timing the new solver

# --- Original SOR Kernel (kept for reference/alternative backend) ---
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
                    term_x = (phi[i + 1, j, k] + phi[i - 1, j, k]) * dx2_inv
                    term_y = (phi[i, j + 1, k] + phi[i, j - 1, k]) * dy2_inv
                    term_z = (phi[i, j, k + 1] + phi[i, j, k - 1]) * dz2_inv 
                    
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


# --- Helper for applying BCs to phi (used by both backends) ---
def _apply_phi_boundary_conditions(phi, processed_bcs, nx_total, ny_total, nz_total):
    """
    Applies boundary conditions to the phi field (ghost cells).
    This function should be called before the iterative solver and possibly after each iteration
    if Neumann BCs are strictly enforced in a non-matrix solver, or once after a matrix solver.
    """
    for bc_name, bc in processed_bcs.items():
        bc_type = bc.get("type")
        apply_to_fields = bc.get("apply_to", [])
            
        # Handle Dirichlet boundary conditions for phi (where pressure is specified, often phi=0)
        # For pressure projection, phi represents a pressure *correction*.
        # If pressure is fixed (Dirichlet), the correction at that boundary is often zero.
        if bc_type == "dirichlet" and "pressure" in apply_to_fields:
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)
            target_value_for_phi = 0.0 

            if ghost_indices.size > 0:
                valid_mask = (
                    (ghost_indices[:, 0] >= 0) & (ghost_indices[:, 0] < nx_total) &
                    (ghost_indices[:, 1] >= 0) & (ghost_indices[:, 1] < ny_total) &
                    (ghost_indices[:, 2] >= 0) & (ghost_indices[:, 2] < nz_total)
                )
                safe_indices = ghost_indices[valid_mask]
                if safe_indices.size > 0:
                    phi[safe_indices[:, 0], safe_indices[:, 1], safe_indices[:, 2]] = target_value_for_phi
            
        # Handle Neumann boundary conditions for phi (where velocity is specified as Dirichlet/fixed)
        # For zero normal velocity at a boundary (e.g., no-slip walls), the normal derivative of phi is zero (∂phi/∂n = 0).
        # This translates to phi[ghost_cell] = phi[adjacent_interior_cell].
        elif bc_type == "dirichlet" and "velocity" in apply_to_fields:
            cell_indices = np.array(bc.get("cell_indices", []), dtype=int)       # Interior cells adjacent to boundary
            ghost_indices = np.array(bc.get("ghost_indices", []), dtype=int)      # Ghost cells for this boundary

            if cell_indices.size > 0 and ghost_indices.size > 0 and cell_indices.shape == ghost_indices.shape:
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
                    phi[safe_ghost_indices[:, 0], safe_ghost_indices[:, 1], safe_ghost_indices[:, 2]] = \
                        phi[safe_cell_indices[:, 0], safe_cell_indices[:, 1], safe_cell_indices[:, 2]]


# --- Matrix Assembly for BiCGSTAB ---
def _assemble_poisson_matrix(nx, ny, nz, dx, dy, dz, processed_bcs):
    """
    Assembles the sparse matrix for the Poisson equation (∇²phi).
    This function constructs the linear system A * phi_flat = b_flat.
    The matrix A includes the discretization of the Laplacian and implicitly
    handles boundary conditions for phi.
    Args:
        nx, ny, nz (int): Dimensions of the *interior* grid.
        dx, dy, dz (float): Grid spacing.
        processed_bcs (dict): Dictionary of boundary conditions.

    Returns:
        scipy.sparse.csr_matrix: The assembled sparse matrix A.
    """
    total_cells = nx * ny * nz
    A = lil_matrix((total_cells, total_cells), dtype=np.float64)

    dx2 = dx * dx
    dy2 = dy * dy
    dz2 = dz * dz

    # Coefficients for the 7-point stencil for the Laplacian
    # 1/dx^2, 1/dy^2, 1/dz^2 on off-diagonals
    # -2(1/dx^2 + 1/dy^2 + 1/dz^2) on main diagonal
    
    # Map (i, j, k) to a single index for the flattened system
    def to_flat_idx(i, j, k):
        return i + j * nx + k * nx * ny

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                current_idx = to_flat_idx(i, j, k)
                
                # Main diagonal coefficient
                diag_coeff = -2.0 * (1.0/dx2 + 1.0/dy2 + 1.0/dz2)
                
                # --- Handle neighbors and boundary conditions ---
                # This part is crucial for correctly incorporating BCs into the matrix.
                # The logic below assumes a standard 7-point stencil and will modify
                # the diagonal based on how many "real" neighbors exist, and modify
                # the off-diagonal terms for internal connections.

                # By default, add contributions from all 6 neighbors.
                # We'll adjust `diag_coeff` and add/remove off-diagonal entries
                # based on the boundary conditions.
                
                # For `pressure` Dirichlet BCs:
                # If a cell (i,j,k) is adjacent to a boundary where pressure is fixed (meaning phi=0),
                # the term for that ghost cell (e.g., phi_ghost / dx^2) drops out of the equation.
                # This means the corresponding off-diagonal entry in A is zero, and the diagonal
                # coefficient needs to be adjusted (less negative, as it loses a -1/dx^2 part).

                # For `velocity` Dirichlet BCs (which imply Neumann for phi: d(phi)/dn = 0):
                # If d(phi)/dn = 0, then phi_ghost = phi_interior_adjacent.
                # For an interior cell (i,j,k) next to a boundary, say i=0 (left face),
                # the term (phi_ghost - 2*phi_interior + phi_right)/dx^2 becomes
                # (phi_interior - 2*phi_interior + phi_right)/dx^2 = (-phi_interior + phi_right)/dx^2
                # This means the diagonal coefficient for phi_interior becomes less negative (e.g., -1/dx^2 instead of -2/dx^2),
                # and the off-diagonal term for phi_right remains 1/dx^2.

                # Initialize with full 7-point stencil coefficients
                A[current_idx, current_idx] = diag_coeff # Diagonal for central cell
                
                # X-direction
                if i > 0: # Left neighbor
                    A[current_idx, to_flat_idx(i - 1, j, k)] = 1.0 / dx2
                else: # Left boundary (i=0) - modify for Neumann or Dirichlet pressure
                    # If this is a wall (velocity BC, so phi Neumann): phi[-1] = phi[0], so (phi[0] - 2phi[0] + phi[1])/dx^2 = (-phi[0]+phi[1])/dx^2
                    # The -2/dx^2 becomes -1/dx^2 (diag_coeff adjustment), and 1/dx^2 to phi[1]
                    # If this is a pressure inlet/outlet (Dirichlet pressure, so phi=0): phi[-1]=0, so (-2phi[0]+phi[1])/dx^2
                    # The -2/dx^2 remains, and 1/dx^2 to phi[1]
                    # This logic needs to inspect `processed_bcs` to determine the specific BC.
                    # For a robust implementation, you'd iterate over `processed_bcs` and modify `A` specifically.
                    # As a general rule, if `dx` neighbor is a Neumann boundary, `diag_coeff` loses a `1/dx2` contribution.
                    # If `dx` neighbor is a Dirichlet boundary, `diag_coeff` and RHS account for `phi_b = 0`.
                    pass # Handled by specific BC logic below
                
                if i < nx - 1: # Right neighbor
                    A[current_idx, to_flat_idx(i + 1, j, k)] = 1.0 / dx2
                else: # Right boundary (i=nx-1)
                    pass

                # Y-direction
                if j > 0: # Back neighbor
                    A[current_idx, to_flat_idx(i, j - 1, k)] = 1.0 / dy2
                else: # Back boundary (j=0)
                    pass

                if j < ny - 1: # Front neighbor
                    A[current_idx, to_flat_idx(i, j + 1, k)] = 1.0 / dy2
                else: # Front boundary (j=ny-1)
                    pass

                # Z-direction
                if k > 0: # Bottom neighbor
                    A[current_idx, to_flat_idx(i, j, k - 1)] = 1.0 / dz2
                else: # Bottom boundary (k=0)
                    pass

                if k < nz - 1: # Top neighbor
                    A[current_idx, to_flat_idx(i, j, k + 1)] = 1.0 / dz2
                else: # Top boundary (k=nz-1)
                    pass
                
                # --- Specific BC Modifications to A (based on mesh_info) ---
                # Iterate through boundary conditions to adjust matrix A.
                # This is a conceptual implementation and needs careful mapping of face_ids/labels
                # to (i,j,k) coordinates and the corresponding matrix rows/columns.

                # Helper to check if a cell (i,j,k) is on a specific boundary face (interior coordinates)
                def is_on_face(idx, face_id, nx, ny, nz):
                    # Face IDs mapping from input:
                    # 1: x_min (i=0)
                    # 2: x_max (i=nx-1)
                    # 3: y_min (j=0)
                    # 4: y_max (j=ny-1)
                    # 5: z_min (k=0)
                    # 6: z_max (k=nz-1)
                    i, j, k = idx
                    if face_id == 1 and i == 0: return True
                    if face_id == 2 and i == nx - 1: return True
                    if face_id == 3 and j == 0: return True
                    if face_id == 4 and j == ny - 1: return True
                    if face_id == 5 and k == 0: return True
                    if face_id == 6 and k == nz - 1: return True
                    return False

                current_cell_interior_coords = (i, j, k)

                for bc_label, bc_data in processed_bcs.items():
                    bc_faces = bc_data.get("faces", [])
                    bc_type = bc_data.get("type")
                    apply_to_fields = bc_data.get("apply_to", [])

                    for face_id in bc_faces:
                        if is_on_face(current_cell_interior_coords, face_id, nx, ny, nz):
                            if bc_type == "dirichlet" and "pressure" in apply_to_fields:
                                # Dirichlet pressure on face_id implies phi=0 at ghost cells.
                                # The equation for the boundary-adjacent interior cell (i,j,k) loses the term
                                # connecting to the ghost cell. The diagonal `diag_coeff` needs to be adjusted.
                                # Example: for face_id 1 (x_min, i=0), the A[current_idx, to_flat_idx(-1,j,k)] is removed.
                                # The -2/dx^2 in diag_coeff becomes -1/dx^2.
                                if face_id == 1: # x_min (i=0)
                                    A[current_idx, to_flat_idx(i - 1, j, k)] = 0.0 # Ensure no connection to non-existent ghost
                                    A[current_idx, current_idx] += 1.0/dx2 # Adjust diagonal: -2 -> -1 contribution from x-direction
                                elif face_id == 2: # x_max (i=nx-1)
                                    A[current_idx, to_flat_idx(i + 1, j, k)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dx2
                                elif face_id == 3: # y_min (j=0)
                                    A[current_idx, to_flat_idx(i, j - 1, k)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dy2
                                elif face_id == 4: # y_max (j=ny-1)
                                    A[current_idx, to_flat_idx(i, j + 1, k)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dy2
                                elif face_id == 5: # z_min (k=0)
                                    A[current_idx, to_flat_idx(i, j, k - 1)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dz2
                                elif face_id == 6: # z_max (k=nz-1)
                                    A[current_idx, to_flat_idx(i, j, k + 1)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dz2
                                
                                # Note: If multiple BCs apply to a corner/edge, this logic needs refinement
                                # to ensure coefficients are adjusted correctly and not double-counted.
                                # For simplicity, we assume non-overlapping primary BCs for now.

                            elif bc_type == "dirichlet" and "velocity" in apply_to_fields:
                                # Dirichlet velocity on face_id implies Neumann for phi (dphi/dn = 0).
                                # This means ghost_value = interior_adjacent_value.
                                # The stencil changes from (phi_ghost - 2phi_i + phi_neighbor) to (-phi_i + phi_neighbor).
                                # So, the diagonal coefficient (originally -2/dx^2) effectively becomes -1/dx^2.
                                if face_id == 1: # x_min (i=0)
                                    A[current_idx, to_flat_idx(i - 1, j, k)] = 0.0 # No explicit ghost connection
                                    A[current_idx, current_idx] += 1.0/dx2 # Adjust diagonal: -2 -> -1 contribution from x-dir
                                elif face_id == 2: # x_max (i=nx-1)
                                    A[current_idx, to_flat_idx(i + 1, j, k)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dx2
                                elif face_id == 3: # y_min (j=0)
                                    A[current_idx, to_flat_idx(i, j - 1, k)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dy2
                                elif face_id == 4: # y_max (j=ny-1)
                                    A[current_idx, to_flat_idx(i, j + 1, k)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dy2
                                elif face_id == 5: # z_min (k=0)
                                    A[current_idx, to_flat_idx(i, j, k - 1)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dz2
                                elif face_id == 6: # z_max (k=nz-1)
                                    A[current_idx, to_flat_idx(i, j, k + 1)] = 0.0
                                    A[current_idx, current_idx] += 1.0/dz2

    return A.tocsr() # Convert to CSR format for efficient matrix-vector products


# --- RHS Vector Construction for BiCGSTAB ---
def _calculate_rhs_vector(divergence, mesh_info, time_step, processed_bcs):
    """
    Constructs the right-hand side vector for the Poisson equation.
    b_flat = - (divergence / time_step).
    This vector corresponds to the flattened interior cells.
    Contributions from non-zero Dirichlet phi boundary conditions also go here.
    """
    nx_interior, ny_interior, nz_interior = mesh_info['grid_shape']
    
    # Initialize rhs_flat with the divergence values
    rhs_flat = (-divergence / time_step).flatten()

    dx2 = mesh_info["dx"] * mesh_info["dx"]
    dy2 = mesh_info["dy"] * mesh_info["dy"]
    dz2 = mesh_info["dz"] * mesh_info["dz"]

    def to_flat_idx(i, j, k):
        return i + j * nx_interior + k * nx_interior * ny_interior

    # Add contributions from Dirichlet pressure boundaries (if phi is not zero there)
    # In the pressure correction scheme, typically phi=0 at Dirichlet pressure boundaries,
    # so this section often just asserts that or handles rare cases of non-zero phi.
    # The current `target_value_for_phi = 0.0` in `_apply_phi_boundary_conditions` implies this.
    # If the fixed pressure was P_fixed, then phi at boundary ghost cells would be (P_fixed - P_current)
    # but for correction, phi is often zero.

    for bc_label, bc_data in processed_bcs.items():
        bc_faces = bc_data.get("faces", [])
        bc_type = bc_data.get("type")
        apply_to_fields = bc_data.get("apply_to", [])

        if bc_type == "dirichlet" and "pressure" in apply_to_fields:
            # Here, we'd iterate over interior cells adjacent to Dirichlet pressure boundaries
            # and add (phi_boundary / dx^2) * (-1) to the RHS for those cells.
            # Since `target_value_for_phi` is 0.0, this term will be 0.
            # If `target_value_for_phi` (the correction at the boundary) was non-zero,
            # it would contribute to the RHS.
            # For your current setup, it's likely no change needed here.
            pass

    return rhs_flat


# --- Main Solve Function ---
def solve_poisson_for_phi(divergence, mesh_info, time_step,
                          omega=1.7, max_iterations=1000, tolerance=1e-6,
                          return_residual=False, backend="bicgstab", preconditioner_type="ilu"):
    """
    Solves the Poisson equation for the pressure correction potential (phi).
    ∇²phi = (1/dt) * (∇·u*)

    Args:
        divergence (np.ndarray): The divergence of the tentative velocity field (∇·u*),
                                 shape (nx, ny, nz) (interior cells only).
        mesh_info (dict): Grid metadata including 'grid_shape' (interior nx, ny, nz)
                          and 'dx', 'dy', 'dz', and 'boundary_conditions'.
        time_step (float): The current simulation time step (dt).
        omega (float): Relaxation factor for SOR (only used for 'sor' backend).
        max_iterations (int): Maximum iterations for the solver.
        tolerance (float): Convergence tolerance for the solver.
        return_residual (bool): If True, returns (phi, residual), else just phi.
        backend (str): Solver backend ("sor" or "bicgstab").
        preconditioner_type (str): Preconditioner for 'bicgstab' ("none" or "ilu").

    Returns:
        np.ndarray or tuple: The solved phi field (with ghost cells)
                              or (phi, final_residual) if return_residual is True.
    """
    
    # Get interior grid dimensions from mesh_info
    nx_interior, ny_interior, nz_interior = mesh_info['grid_shape']
    # Calculate total grid dimensions including ghost cells
    nx_total, ny_total, nz_total = nx_interior + 2, ny_interior + 2, nz_interior + 2

    dx = mesh_info["dx"]
    dy = mesh_info["dy"]
    dz = mesh_info["dz"]

    processed_bcs = mesh_info.get("boundary_conditions", {})

    final_residual = np.nan # Initialize for return

    if backend == "sor":
        # Initialize phi field with zeros. It will have ghost cells.
        phi = np.zeros((nx_total, ny_total, nz_total), dtype=np.float64)

        # Construct the RHS 'b' for the Poisson equation (∇²phi = b).
        # b = - (divergence / time_step).
        # The 'divergence' input is for interior cells, so place it in the interior of 'rhs'.
        rhs = np.zeros_like(phi)
        rhs[1:-1, 1:-1, 1:-1] = -divergence / time_step

        # Defensive clamping for RHS in case divergence had issues
        rhs_has_nan_before_clamp = np.isnan(rhs).any()
        rhs_has_inf_before_clamp = np.isinf(rhs).any()
        if rhs_has_nan_before_clamp or rhs_has_inf_before_clamp:
            print(f"[Poisson DEBUG] RHS stats BEFORE clamp: min={np.nanmin(rhs):.4e}, max={np.nanmax(rhs):.4e}, has_nan={rhs_has_nan_before_clamp}, has_inf={rhs_has_inf_before_clamp}")
            print("❌ Warning: NaNs or Infs detected in Poisson RHS — clamping to zero.")
        rhs = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[Poisson DEBUG] RHS stats AFTER clamp: min={np.nanmin(rhs):.4e}, max={np.nanmax(rhs):.4e}, has_nan={np.any(np.isnan(rhs))}, has_inf={np.any(np.isinf(rhs))}")

        # Apply boundary conditions to the phi field (ghost cells) - this needs to be done *before* the SOR kernel
        _apply_phi_boundary_conditions(phi, processed_bcs, nx_total, ny_total, nz_total)

        residual_container = np.zeros(1, dtype=np.float64)
        print(f"[Poisson Solver] Starting SOR solver with {max_iterations} iterations and tolerance {tolerance}.")
        
        phi_has_nan_before_sor = np.isnan(phi).any()
        phi_has_inf_before_sor = np.isinf(phi).any()
        if phi_has_nan_before_sor or phi_has_inf_before_sor:
            print(f"[Poisson DEBUG] Phi stats BEFORE SOR kernel: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}, has_nan={phi_has_nan_before_sor}, has_inf={phi_has_inf_before_sor}")
            phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"[Poisson DEBUG] Phi stats AFTER clamping BEFORE SOR kernel: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}")

        phi = _sor_kernel_with_residual(phi, rhs, dx, dy, dz, omega,
                                        float(max_iterations), float(tolerance), residual_container)
        
        print(f"[Poisson Solver] SOR finished. Final residual: {residual_container[0]:.6e}")

        # Apply boundary conditions one last time to the solved phi field
        _apply_phi_boundary_conditions(phi, processed_bcs, nx_total, ny_total, nz_total)

        final_residual = residual_container[0]

    elif backend == "bicgstab":
        print(f"[Poisson Solver] Starting BiCGSTAB solver with {max_iterations} iterations and tolerance {tolerance}.")
        
        # 1. Assemble the matrix A
        start_time_assembly = time.time()
        A = _assemble_poisson_matrix(nx_interior, ny_interior, nz_interior, dx, dy, dz, processed_bcs)
        end_time_assembly = time.time()
        print(f"[Poisson Solver] Matrix assembly took: {end_time_assembly - start_time_assembly:.4f} seconds.")

        # 2. Prepare the RHS vector b
        b_flat = _calculate_rhs_vector(divergence, mesh_info, time_step, processed_bcs)
        
        # Defensive clamping for RHS
        rhs_has_nan_before_clamp = np.isnan(b_flat).any()
        rhs_has_inf_before_clamp = np.isinf(b_flat).any()
        if rhs_has_nan_before_clamp or rhs_has_inf_before_clamp:
            print(f"[Poisson DEBUG] BiCGSTAB RHS stats BEFORE clamp: min={np.nanmin(b_flat):.4e}, max={np.nanmax(b_flat):.4e}, has_nan={rhs_has_nan_before_clamp}, has_inf={rhs_has_inf_before_clamp}")
            print("❌ Warning: NaNs or Infs detected in Poisson RHS for BiCGSTAB — clamping to zero.")
        b_flat = np.nan_to_num(b_flat, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[Poisson DEBUG] BiCGSTAB RHS stats AFTER clamp: min={np.nanmin(b_flat):.4e}, max={np.nanmax(b_flat):.4e}, has_nan={np.any(np.isnan(b_flat))}, has_inf={np.any(np.isinf(b_flat))}")

        # Initialize phi_flat (initial guess for the solver)
        # Often, a zero guess is sufficient, or use previous time step's phi.
        phi_flat_initial_guess = np.zeros(nx_interior * ny_interior * nz_interior, dtype=np.float64)

        # 3. Setup Preconditioner (ILU)
        M = None
        if preconditioner_type == "ilu":
            try:
                start_time_preconditioner = time.time()
                # spilu returns an object that has a .solve() method
                ilu = spilu(A, drop_tol=1e-5) # drop_tol can be tuned, e.g., 1e-4, 1e-3
                M = LinearOperator(A.shape, matvec=ilu.solve)
                end_time_preconditioner = time.time()
                print(f"[Poisson Solver] ILU preconditioner setup took: {end_time_preconditioner - start_time_preconditioner:.4f} seconds.")
            except RuntimeError as e:
                print(f"⚠️ Warning: ILU factorization failed ({e}). This often means the matrix is singular or has issues (e.g., disconnected grid, severe aspect ratios). Proceeding without preconditioner.")
                M = None
            except ValueError as e:
                print(f"⚠️ Warning: ILU setup ValueError ({e}). Proceeding without preconditioner.")
                M = None
        elif preconditioner_type != "none":
            raise ValueError(f"Unsupported preconditioner type '{preconditioner_type}'. Choose 'none' or 'ilu'.")

        # 4. Call BiCGSTAB solver
        start_time_solve = time.time()
        # BiCGSTAB returns the solution vector and an info code (0 for convergence)
        phi_flat, info = bicgstab(A, b_flat, x0=phi_flat_initial_guess,
                                  tol=tolerance, maxiter=max_iterations, M=M)
        end_time_solve = time.time()
        
        if info == 0:
            print(f"[Poisson Solver] BiCGSTAB converged in {end_time_solve - start_time_solve:.4f} seconds.")
            # Calculate final residual for logging (bicgstab doesn't return it directly as a convergence history)
            final_residual = np.linalg.norm(A @ phi_flat - b_flat)
        else:
            print(f"[Poisson Solver] BiCGSTAB did NOT converge! Info code: {info} (failed after {max_iterations} iterations or other issue).")
            # If not converged, solution might be garbage. Clamp to prevent further propagation.
            phi_flat = np.nan_to_num(phi_flat, nan=0.0, posinf=0.0, neginf=0.0) 
            final_residual = np.linalg.norm(A @ phi_flat - b_flat) # Still calculate for logging
            print(f"  > Solution potentially unreliable due to non-convergence. Clamped NaNs/Infs.")


        # Reshape the flattened phi back to a 3D grid with ghost cells
        phi = np.zeros((nx_total, ny_total, nz_total), dtype=np.float64)
        phi[1:-1, 1:-1, 1:-1] = phi_flat.reshape(nx_interior, ny_interior, nz_interior)

        # Re-apply BCs to ghost cells from the solved interior phi (especially for Neumann)
        # This is important because the solver only operates on the interior nodes.
        _apply_phi_boundary_conditions(phi, processed_bcs, nx_total, ny_total, nz_total)

        print(f"[Poisson Solver] BiCGSTAB Final residual (A*x - b): {final_residual:.6e}")

    else:
        raise ValueError(f"Unsupported backend '{backend}'. Choose 'sor' or 'bicgstab'.")

    # DEBUG: Final phi field stats after final BC application
    print(f"[Poisson DEBUG] Phi stats AFTER final BCs: min={np.nanmin(phi):.4e}, max={np.nanmax(phi):.4e}, has_nan={np.any(np.isnan(phi))}, has_inf={np.any(np.isinf(phi))}")

    return (phi, final_residual) if return_residual else phi



