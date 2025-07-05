# src/numerical_methods/poisson_solver.py

import numpy as np
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import spsolve, bicgstab, LinearOperator
from scipy.sparse.linalg import spilu # For ILU preconditioner
import sys

# Constants for solver types
SOLVER_BACKEND_DIRECT = "direct"
SOLVER_BACKEND_BICGSTAB = "bicgstab"

# Preconditioner types
PRECONDITIONER_NONE = "none"
PRECONDITIONER_ILU = "ilu"


def _assemble_poisson_matrix(nx_interior, ny_interior, nz_interior, dx, dy, dz, boundary_conditions):
    """
    Assembles the sparse matrix A for the Poisson equation (Laplacian operator)
    for interior cells, incorporating boundary conditions.

    Args:
        nx_interior (int): Number of interior cells in x-direction.
        ny_interior (int): Number of interior cells in y-direction.
        nz_interior (int): Number of interior cells in z-direction.
        dx (float): Grid spacing in x-direction.
        dy (float): Grid spacing in y-direction.
        dz (float): Grid spacing in z-direction.
        boundary_conditions (dict): Dictionary with processed boundary conditions.
                                    Expected format:
                                    {
                                        'x_min': {'type': 'neumann'/'dirichlet', 'value': 0.0},
                                        'x_max': {'type': 'neumann'/'dirichlet', 'value': 0.0},
                                        ...
                                        'periodic_x': True/False,
                                        'periodic_y': True/False,
                                        'periodic_z': True/False,
                                    }

    Returns:
        scipy.sparse.csr_matrix: The assembled sparse matrix A.
    """
    N = nx_interior * ny_interior * nz_interior
    A = lil_matrix((N, N))

    # Coefficients for the 7-point stencil of the 3D Laplacian (finite difference)
    # ∇²φ = (φ_{i+1} - 2φ_i + φ_{i-1})/dx² + (φ_{j+1} - 2φ_j + φ_{j-1})/dy² + (φ_{k+1} - 2φ_k + φ_{k-1})/dz²
    # The Poisson equation is ∇²φ = RHS, so A*phi = RHS
    # Diagonal term coefficient for central cell (i,j,k)
    C0 = -2.0 / (dx**2) - 2.0 / (dy**2) - 2.0 / (dz**2)
    # Off-diagonal coefficients for neighbors
    Cx = 1.0 / (dx**2)
    Cy = 1.0 / (dy**2)
    Cz = 1.0 / (dz**2)

    # Helper function to convert 3D interior indices to a 1D flattened index
    def to_flat_idx(i, j, k):
        return i + j * nx_interior + k * nx_interior * ny_interior

    for k in range(nz_interior):
        for j in range(ny_interior):
            for i in range(nx_interior):
                current_idx = to_flat_idx(i, j, k)
                
                # Initialize row for current cell
                A[current_idx, current_idx] = C0

                # X-direction neighbors
                # i-1 neighbor
                if i > 0:
                    A[current_idx, to_flat_idx(i - 1, j, k)] = Cx
                elif boundary_conditions.get('periodic_x', False):
                    A[current_idx, to_flat_idx(nx_interior - 1, j, k)] = Cx # Wrap around
                else: # Boundary at x_min
                    # Handle boundary conditions for the leftmost interior cells (i=0)
                    bc_type = boundary_conditions['x_min']['type']
                    if bc_type == 'neumann':
                        # For Neumann, ∂φ/∂n = 0 implies φ_ghost = φ_interior, so φ_i-1 = φ_i
                        # (φ_{i+1} - 2φ_i + φ_{i-1})/dx^2 becomes (φ_{i+1} - φ_i)/dx^2 if φ_i-1 = φ_i
                        # So, -2/dx^2 becomes -1/dx^2 on diagonal (C0 term)
                        # The Cx term is effectively absorbed by the diagonal.
                        A[current_idx, current_idx] += Cx # Adjust diagonal term, effectively making Cx*phi_i on LHS
                    # Dirichlet BCs are handled by modifying the RHS vector, not A, or by reducing matrix size.
                    # For simplicity, if not periodic or Neumann, we assume Dirichlet where ghost cell is known and contributes to RHS.
                    # So, no A[current_idx, to_flat_idx(i - 1, j, k)] term.

                # i+1 neighbor
                if i < nx_interior - 1:
                    A[current_idx, to_flat_idx(i + 1, j, k)] = Cx
                elif boundary_conditions.get('periodic_x', False):
                    A[current_idx, to_flat_idx(0, j, k)] = Cx # Wrap around
                else: # Boundary at x_max
                    bc_type = boundary_conditions['x_max']['type']
                    if bc_type == 'neumann':
                        A[current_idx, current_idx] += Cx # Adjust diagonal term

                # Y-direction neighbors
                # j-1 neighbor
                if j > 0:
                    A[current_idx, to_flat_idx(i, j - 1, k)] = Cy
                elif boundary_conditions.get('periodic_y', False):
                    A[current_idx, to_flat_idx(i, ny_interior - 1, k)] = Cy # Wrap around
                else: # Boundary at y_min
                    bc_type = boundary_conditions['y_min']['type']
                    if bc_type == 'neumann':
                        A[current_idx, current_idx] += Cy

                # j+1 neighbor
                if j < ny_interior - 1:
                    A[current_idx, to_flat_idx(i, j + 1, k)] = Cy
                elif boundary_conditions.get('periodic_y', False):
                    A[current_idx, to_flat_idx(i, 0, k)] = Cy # Wrap around
                else: # Boundary at y_max
                    bc_type = boundary_conditions['y_max']['type']
                    if bc_type == 'neumann':
                        A[current_idx, current_idx] += Cy

                # Z-direction neighbors
                # k-1 neighbor
                if k > 0:
                    A[current_idx, to_flat_idx(i, j, k - 1)] = Cz
                elif boundary_conditions.get('periodic_z', False):
                    A[current_idx, to_flat_idx(i, j, nz_interior - 1)] = Cz # Wrap around
                else: # Boundary at z_min
                    bc_type = boundary_conditions['z_min']['type']
                    if bc_type == 'neumann':
                        A[current_idx, current_idx] += Cz

                # k+1 neighbor
                if k < nz_interior - 1:
                    A[current_idx, to_flat_idx(i, j, k + 1)] = Cz
                elif boundary_conditions.get('periodic_z', False):
                    A[current_idx, to_flat_idx(i, j, 0)] = Cz # Wrap around
                else: # Boundary at z_max
                    bc_type = boundary_conditions['z_max']['type']
                    if bc_type == 'neumann':
                        A[current_idx, current_idx] += Cz
    
    return A.tocsr() # Convert to CSR for efficient arithmetic operations and solvers


def _apply_poisson_rhs_bcs(b_flat, nx_interior, ny_interior, nz_interior, dx, dy, dz, boundary_conditions):
    """
    Applies boundary conditions to the RHS vector 'b' for the Poisson equation.
    This handles contributions from Dirichlet boundary values.

    Args:
        b_flat (np.ndarray): The flattened RHS vector, initially containing only interior divergence.
        nx_interior, ny_interior, nz_interior (int): Interior grid dimensions.
        dx, dy, dz (float): Grid spacing.
        boundary_conditions (dict): Processed boundary conditions.

    Returns:
        np.ndarray: The modified RHS vector 'b_flat' with BC contributions.
    """
    # Coefficients for the Laplacian
    Cx = 1.0 / (dx**2)
    Cy = 1.0 / (dy**2)
    Cz = 1.0 / (dz**2)

    def to_flat_idx(i, j, k):
        return i + j * nx_interior + k * nx_interior * ny_interior

    for k in range(nz_interior):
        for j in range(ny_interior):
            for i in range(nx_interior):
                current_idx = to_flat_idx(i, j, k)

                # Contributions from Dirichlet boundaries
                # x_min boundary (i=0)
                if i == 0 and not boundary_conditions.get('periodic_x', False):
                    if boundary_conditions['x_min']['type'] == 'dirichlet':
                        b_flat[current_idx] -= Cx * boundary_conditions['x_min']['value']
                
                # x_max boundary (i=nx_interior-1)
                if i == nx_interior - 1 and not boundary_conditions.get('periodic_x', False):
                    if boundary_conditions['x_max']['type'] == 'dirichlet':
                        b_flat[current_idx] -= Cx * boundary_conditions['x_max']['value']
                
                # y_min boundary (j=0)
                if j == 0 and not boundary_conditions.get('periodic_y', False):
                    if boundary_conditions['y_min']['type'] == 'dirichlet':
                        b_flat[current_idx] -= Cy * boundary_conditions['y_min']['value']
                
                # y_max boundary (j=ny_interior-1)
                if j == ny_interior - 1 and not boundary_conditions.get('periodic_y', False):
                    if boundary_conditions['y_max']['type'] == 'dirichlet':
                        b_flat[current_idx] -= Cy * boundary_conditions['y_max']['value']

                # z_min boundary (k=0)
                if k == 0 and not boundary_conditions.get('periodic_z', False):
                    if boundary_conditions['z_min']['type'] == 'dirichlet':
                        b_flat[current_idx] -= Cz * boundary_conditions['z_min']['value']
                
                # z_max boundary (k=nz_interior-1)
                if k == nz_interior - 1 and not boundary_conditions.get('periodic_z', False):
                    if boundary_conditions['z_max']['type'] == 'dirichlet':
                        b_flat[current_idx] -= Cz * boundary_conditions['z_max']['value']
    return b_flat


def solve_poisson_for_phi(
    divergence_field: np.ndarray,
    mesh_info: dict,
    dt: float,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    backend: str = SOLVER_BACKEND_BICGSTAB, # Default to BiCGSTAB
    preconditioner_type: str = PRECONDITIONER_ILU, # Default to ILU
    return_residual: bool = False
) -> tuple[np.ndarray, float]:
    """
    Solves the Poisson equation for pressure correction potential (phi).

    The equation solved is: ∇²φ = (ρ/Δt) * ∇·u
    where ∇·u is the divergence of the tentative velocity field.

    Args:
        divergence_field (np.ndarray): The divergence of the tentative velocity field,
                                       including ghost cells. Shape (nx+2, ny+2, nz+2).
        mesh_info (dict): Dictionary with mesh details including 'grid_shape', 'dx', 'dy', 'dz',
                          and boundary condition info.
        dt (float): Time step size.
        tolerance (float): Solver tolerance.
        max_iterations (int): Maximum number of iterations for iterative solvers.
        backend (str): Which solver backend to use ("direct" or "bicgstab").
        preconditioner_type (str): Type of preconditioner for iterative solvers ("none" or "ilu").
        return_residual (bool): If True, returns the final residual of the iterative solver.

    Returns:
        tuple[np.ndarray, float]: The pressure correction potential (phi) field
                                  (shape nx+2, ny+2, nz+2) and the final residual (if return_residual=True).
                                  If return_residual is False, only phi is returned.
    """
    nx_total, ny_total, nz_total = divergence_field.shape
    nx_interior, ny_interior, nz_interior = mesh_info['grid_shape'] # This should be (nx, ny, nz) without ghost cells

    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Slice for interior cells
    interior_slice = (slice(1, nx_interior + 1), slice(1, ny_interior + 1), slice(1, nz_interior + 1))

    # Construct RHS vector b = (rho / dt) * div_u for interior cells
    # Note: density is typically absorbed into the pressure, so P_new = P_old + phi
    # If the velocity correction is -dt/rho * grad(phi), then the Poisson equation is div(grad(phi)) = (rho/dt) * div(u_star)
    # So, RHS = (rho/dt) * divergence_field_interior
    # For simplicity, we are solving for phi, such that the final pressure P = P_old + phi.
    # The pressure correction step then involves P_new = P_old + phi.
    # The velocity correction will be (v_star - grad_phi*dt). No rho. This means phi has units of pressure.
    # If using the standard projection method ( Chorin's method), the Poisson equation is:
    # ∇²φ = (1/Δt) * ∇·u*
    # So, the RHS is (1/dt) * divergence_field_interior
    
    # We will use RHS = divergence_field_interior / dt, assuming phi directly relates to pressure change.
    # The density factor is typically handled in `apply_pressure_correction`.
    rhs_interior = divergence_field[interior_slice] / dt
    b_flat = rhs_interior.flatten() # Flatten for solver

    # Extract processed boundary conditions from mesh_info
    # Default to non-periodic and all-neumann if not specified for robustness
    boundary_conditions = mesh_info.get('boundary_conditions', {})
    default_bc_info = {'type': 'neumann', 'value': 0.0} # Default to zero Neumann flux

    processed_bcs = {
        'x_min': boundary_conditions.get('x_min', default_bc_info),
        'x_max': boundary_conditions.get('x_max', default_bc_info),
        'y_min': boundary_conditions.get('y_min', default_bc_info),
        'y_max': boundary_conditions.get('y_max', default_bc_info),
        'z_min': boundary_conditions.get('z_min', default_bc_info),
        'z_max': boundary_conditions.get('z_max', default_bc_info),
        'periodic_x': boundary_conditions.get('periodic_x', False),
        'periodic_y': boundary_conditions.get('periodic_y', False),
        'periodic_z': boundary_conditions.get('periodic_z', False),
    }

    # Assemble the Poisson matrix A for interior cells
    A = _assemble_poisson_matrix(nx_interior, ny_interior, nz_interior, dx, dy, dz, processed_bcs)

    # Apply Dirichlet boundary conditions to the RHS vector 'b'
    b_flat = _apply_poisson_rhs_bcs(b_flat, nx_interior, ny_interior, nz_interior, dx, dy, dz, processed_bcs)

    phi_flat = None
    final_residual = np.nan

    if backend == SOLVER_BACKEND_DIRECT:
        print("[Poisson Solver] Using direct solver (spsolve).")
        try:
            phi_flat = spsolve(A, b_flat)
            final_residual = np.linalg.norm(A @ phi_flat - b_flat) # Calculate residual for direct solver
        except Exception as e:
            print(f"Error in direct Poisson solve: {e}", file=sys.stderr)
            raise
    elif backend == SOLVER_BACKEND_BICGSTAB:
        print(f"[Poisson Solver] Starting BiCGSTAB solver with {max_iterations} iterations and tolerance {tolerance:.1e}.")
        M = None
        if preconditioner_type == PRECONDITIONER_ILU:
            print("[Poisson Solver] Using ILU preconditioner.")
            try:
                M = spilu(A.tocsc(), drop_tol=1e-5, fill_factor=20) # Convert to CSC for spilu
                M_inv = LinearOperator(A.shape, matvec=M.solve)
            except RuntimeError as e: # spilu can raise RuntimeError if matrix is too ill-conditioned
                print(f"WARNING: ILU preconditioning failed: {e}. Attempting without preconditioner.", file=sys.stderr)
                M_inv = None
            except Exception as e:
                print(f"Error setting up ILU preconditioner: {e}. Attempting without preconditioner.", file=sys.stderr)
                M_inv = None
        elif preconditioner_type == PRECONDITIONER_NONE:
            print("[Poisson Solver] No preconditioner used.")
            M_inv = None
        else:
            print(f"WARNING: Unknown preconditioner type '{preconditioner_type}'. Proceeding without preconditioner.", file=sys.stderr)
            M_inv = None

        try:
            # bicgstab returns (x, info) where info = 0 if successful
            phi_flat, info = bicgstab(A, b_flat, tol=tolerance, maxiter=max_iterations, M=M_inv)
            
            if info > 0:
                print(f"WARNING: BiCGSTAB did not converge after {info} iterations. Final residual might be high.", file=sys.stderr)
            elif info < 0:
                print(f"ERROR: BiCGSTAB failed with error code {info}. Invalid input or breakdown.", file=sys.stderr)
                raise RuntimeError(f"BiCGSTAB solver failed with code {info}")
            
            # Calculate the actual final residual
            final_residual = np.linalg.norm(A @ phi_flat - b_flat)
            print(f"[Poisson Solver] BiCGSTAB finished. Info: {info}, Final Residual: {final_residual:.6e}")

        except Exception as e:
            print(f"Error in iterative Poisson solve (BiCGSTAB): {e}", file=sys.stderr)
            raise
    else:
        raise ValueError(f"Unknown Poisson solver backend: {backend}. Choose 'direct' or 'bicgstab'.")

    if phi_flat is None:
        raise RuntimeError("Poisson solver failed to produce a solution (phi_flat is None).")

    # Reshape phi_flat back to 3D interior grid
    phi_interior = phi_flat.reshape((nx_interior, ny_interior, nz_interior))

    # Initialize full phi_field with zeros (including ghost cells)
    phi_field = np.zeros((nx_total, ny_total, nz_total), dtype=phi_interior.dtype)
    phi_field[interior_slice] = phi_interior

    # Apply boundary conditions to phi_field (ghost cells)
    # For Poisson pressure solver, Neumann BC (dPhi/dn = 0) implies ghost cell = interior cell.
    # Dirichlet BC implies ghost cell = 2*BC_value - interior cell (if central difference).
    # This needs to be carefully matched with how A and b were constructed.
    # If the BCs were incorporated into A and b, then phi_field from the solver
    # already satisfies the ghost cell relationships implicitly.
    # For now, let's assume Neumann (zero gradient) implies simple copy for ghost cells
    # or that the system implicitly handles them.
    # A more robust approach would be to calculate ghost cells explicitly here based on BCs.

    # Simple ghost cell update for phi based on interior.
    # For zero Neumann gradient at a wall, phi_ghost = phi_interior_adjacent_to_wall
    # For Dirichlet, phi_ghost values might be explicitly set or derived.
    # If your `apply_boundary_conditions` can handle `phi` as a scalar field, use it.
    # For Poisson, pressure BCs are typically on the gradient, linking to velocity BCs.
    # For a free surface, P=0 Dirichlet can be set. For walls, dP/dn = 0 (Neumann).
    # Since this is a pressure *correction* potential, its BCs are derived from velocity BCs.
    # E.g., if velocity is no-slip at a wall, dP/dn = rho * a_n. For zero velocity divergence, dP/dn = 0.
    # So, zero Neumann for phi is a common assumption.

    # For zero Neumann conditions (most common for pressure correction at solid walls)
    # phi_ghost = phi_interior_adjacent_to_wall
    # This is implicitly handled if the matrix A enforces these conditions.
    # For explicit setting:
    if not processed_bcs.get('periodic_x', False):
        if processed_bcs['x_min']['type'] == 'neumann':
            phi_field[0, :, :] = phi_field[1, :, :]
        if processed_bcs['x_max']['type'] == 'neumann':
            phi_field[nx_total - 1, :, :] = phi_field[nx_total - 2, :, :]
    
    if not processed_bcs.get('periodic_y', False):
        if processed_bcs['y_min']['type'] == 'neumann':
            phi_field[:, 0, :] = phi_field[:, 1, :]
        if processed_bcs['y_max']['type'] == 'neumann':
            phi_field[:, ny_total - 1, :] = phi_field[:, ny_total - 2, :]

    if not processed_bcs.get('periodic_z', False):
        if processed_bcs['z_min']['type'] == 'neumann':
            phi_field[:, :, 0] = phi_field[:, :, 1]
        if processed_bcs['z_max']['type'] == 'neumann':
            phi_field[:, :, nz_total - 1] = phi_field[:, :, nz_total - 2]
    
    # For periodic boundaries, ghost cells are set by `apply_boundary_conditions` later
    # based on the opposite side.
    
    if return_residual:
        return phi_field, final_residual
    else:
        return phi_field



