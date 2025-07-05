# src/numerical_methods/implicit_solver.py

import numpy as np
import sys
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import spsolve # This is for solving the implicit diffusion step

# Import the individual numerical methods and the boundary conditions module.
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term # This module is likely used for explicit diffusion terms if any, or just for conceptual separation
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    from .pressure_correction import apply_pressure_correction
    from physics.boundary_conditions_applicator import apply_boundary_conditions
except ImportError as e:
    print(f"Error importing components for implicit_solver: {e}", file=sys.stderr)
    print("Please ensure all necessary files exist in their respective directories "
          "and contain the expected functions.", file=sys.stderr)
    sys.exit(1)


class ImplicitSolver:
    """
    Implements a semi-implicit (fractional step) solver for the Navier-Stokes equations.
    Diffusion terms are handled implicitly, while advection and pressure gradient terms are explicit.
    """
    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        """
        Intializes the ImplicitSolver.

        Args:
            fluid_properties (dict): Dictionary with fluid density ('rho') and viscosity ('nu').
            mesh_info (dict): Dictionary with mesh details including 'grid_shape', 'dx', 'dy', 'dz',
                              and boundary condition indices.
            dt (float): Time step size.
        """
        print("Initializing ImplicitSolver...")
        self.fluid_properties_dict = fluid_properties # Stored as dict to pass to BCs
        self.density = fluid_properties["density"]
        self.viscosity = fluid_properties["viscosity"]
        self.mesh_info = mesh_info
        self.dt = dt # This dt will be the maximum allowed, actual dt will be dynamic

        # Note: self.nx_int, self.ny_int, self.nz_int from mesh_info['grid_shape']
        # should be the *total* dimensions (including ghost cells) if that's how
        # the mesh is defined consistently across the project.
        # Based on previous debugging, it seems mesh_info['grid_shape'] is the total grid size.
        self.nx_total, self.ny_total, self.nz_total = self.mesh_info['grid_shape']
        self.dx = self.mesh_info['dx']
        self.dy = self.mesh_info['dy']
        self.dz = self.mesh_info['dz']

        # Pre-build the LHS matrix for the implicit diffusion step
        # This matrix is constant as long as dt and mesh_info don't change
        # If dt could change dynamically during simulation, this matrix would need to be rebuilt or updated.
        self.diffusion_matrix_LHS = self._build_diffusion_matrix()
        print("ImplicitSolver initialized.")

    def _build_diffusion_matrix(self) -> lil_matrix:
        """
        Builds the LHS matrix for the implicit diffusion equation (similar to Poisson, but for diffusion).
        The equation for implicit diffusion is:
        u_new - nu * dt * (d2u/dx2 + d2u/dy2 + d2u/dz2) = u_old + dt * RHS_explicit

        Rearranging for implicit solve (u_new):
        (I - nu * dt * Laplacian) * u_new = u_old + dt * RHS_explicit

        So, the LHS matrix is (I - nu * dt * Laplacian)
        """
        # For the diffusion matrix, we are solving for the interior velocities.
        # So, the matrix dimensions should correspond to the number of interior cells.
        # Assuming 1 ghost cell layer on each side, interior dimensions are total - 2.
        nx_interior = self.nx_total - 2
        ny_interior = self.ny_total - 2
        nz_interior = self.nz_total - 2

        N = nx_interior * ny_interior * nz_interior
        LHS_matrix = lil_matrix((N, N))

        # Identity matrix part (for the u_new term)
        LHS_matrix.setdiag(1.0)

        # Laplacian part (nu * dt * Laplacian)
        nu_dt = self.viscosity * self.dt 

        # Coefficients for finite difference Laplacian (negative for diffusion operator)
        coeff_x = nu_dt / (self.dx ** 2)
        coeff_y = nu_dt / (self.dy ** 2)
        coeff_z = nu_dt / (self.dz ** 2)

        # Map (i, j, k) interior index to a single flattened index
        def to_flat_idx(i, j, k):
            return i + j * nx_interior + k * nx_interior * ny_interior

        # Iterate over all interior cells to build the matrix
        for k in range(nz_interior):
            for j in range(ny_interior):
                for i in range(nx_interior):
                    idx = to_flat_idx(i, j, k)

                    # Diagonal term: 1 (from Identity) + 2*coeff_x + 2*coeff_y + 2*coeff_z (from -nu*dt*Laplacian)
                    LHS_matrix[idx, idx] += 2 * coeff_x + 2 * coeff_y + 2 * coeff_z

                    # Off-diagonal terms: -coeff_x, -coeff_y, -coeff_z for neighbors
                    # x-direction neighbors
                    if i > 0:
                        LHS_matrix[idx, to_flat_idx(i - 1, j, k)] = -coeff_x
                    if i < nx_interior - 1:
                        LHS_matrix[idx, to_flat_idx(i + 1, j, k)] = -coeff_x

                    # y-direction neighbors
                    if j > 0:
                        LHS_matrix[idx, to_flat_idx(i, j - 1, k)] = -coeff_y
                    if j < ny_interior - 1:
                        LHS_matrix[idx, to_flat_idx(i, j + 1, k)] = -coeff_y

                    # z-direction neighbors
                    if k > 0:
                        LHS_matrix[idx, to_flat_idx(i, j, k - 1)] = -coeff_z
                    if k < nz_interior - 1:
                        LHS_matrix[idx, to_flat_idx(i, j, k + 1)] = -coeff_z
                
        return LHS_matrix.tocsr() # Convert to CSR for efficient sparse solve


    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs one semi-implicit time step, using implicit diffusion.

        Args:
            velocity_field (np.ndarray): Current velocity field (U, V, W components).
                                         Shape: (nx_total, ny_total, nz_total, 3).
            pressure_field (np.ndarray): Current pressure field. Shape: (nx_total, ny_total, nz_total).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Updated velocity_field, pressure_field, and the divergence field after correction.
        """
        
        # Create copies to work with, ensuring original fields are not modified until step is complete
        current_velocity = np.copy(velocity_field)
        current_pressure = np.copy(pressure_field)
        
        num_pseudo_iterations = 2 # Or 1, depending on desired accuracy vs. performance
        poisson_tolerance = 1e-6  # Default tolerance for Poisson solver
        poisson_max_iter = 1000   # Default max iterations for Poisson solver

        # Derive interior dimensions from total dimensions (total - 2 ghost cells on each side)
        nx_interior = self.nx_total - 2
        ny_interior = self.ny_total - 2
        nz_interior = self.nz_total - 2
        
        # Define slice for interior cells (excluding ghost cells)
        interior_slice_3d = (slice(1, self.nx_total - 1), slice(1, self.ny_total - 1), slice(1, self.nz_total - 1))
        
        divergence_after_correction_field = np.zeros_like(current_pressure)

        print(f"[DEBUG] Initial velocity in step(): min={np.nanmin(current_velocity):.4e}, max={np.nanmax(current_velocity):.4e}, has_nan={np.any(np.isnan(current_velocity))}, has_inf={np.any(np.isinf(current_velocity))}")
        print(f"[DEBUG] Initial pressure in step(): min={np.nanmin(current_pressure):.4e}, max={np.nanmax(current_pressure):.4e}, has_nan={np.any(np.isnan(current_pressure))}, has_inf={np.any(np.isinf(current_pressure))}")

        if self.dt <= 0:
            raise ValueError("Time step (self.dt) is zero or negative. Cannot proceed with simulation.")

        for pseudo_iter in range(num_pseudo_iterations):
            print(f"\n--- Implicit Solver Pseudo-Iteration {pseudo_iter + 1}/{num_pseudo_iterations} ---")
            
            # --- Extract INTERIOR velocity components for operations ---
            u_interior = current_velocity[interior_slice_3d + (0,)]
            v_interior = current_velocity[interior_slice_3d + (1,)]
            w_interior = current_velocity[interior_slice_3d + (2,)]

            u_flat_interior = u_interior.flatten()
            v_flat_interior = v_interior.flatten()
            w_flat_interior = w_interior.flatten()

            # --- Explicit Advection Term ---
            advection_u_interior = compute_advection_term(current_velocity[..., 0], current_velocity, self.mesh_info)
            advection_v_interior = compute_advection_term(current_velocity[..., 1], current_velocity, self.mesh_info)
            advection_w_interior = compute_advection_term(current_velocity[..., 2], current_velocity, self.mesh_info)
            
            print(f"[DEBUG] Advection_u stats: min={np.nanmin(advection_u_interior):.4e}, max={np.nanmax(advection_u_interior):.4e}, has_nan={np.any(np.isnan(advection_u_interior))}, has_inf={np.any(np.isinf(advection_u_interior))}")
            print(f"[DEBUG] Advection_v stats: min={np.nanmin(advection_v_interior):.4e}, max={np.nanmax(advection_v_interior):.4e}, has_nan={np.any(np.isnan(advection_v_interior))}, has_inf={np.any(np.isinf(advection_v_interior))}")
            print(f"[DEBUG] Advection_w stats: min={np.nanmin(advection_w_interior):.4e}, max={np.nanmax(advection_w_interior):.4e}, has_nan={np.any(np.isnan(advection_w_interior))}, has_inf={np.any(np.isinf(advection_w_interior))}")

            # Flatten advection terms for linear system RHS
            advection_u_flat = advection_u_interior.flatten()
            advection_v_flat = advection_v_interior.flatten()
            advection_w_flat = advection_w_interior.flatten()

            # --- Pressure Gradient Term (Explicit) ---
            grad_p_x = np.zeros_like(current_pressure)
            grad_p_y = np.zeros_like(current_pressure)
            grad_p_z = np.zeros_like(current_pressure)

            # Central difference for interior cells
            grad_p_x[1:-1, 1:-1, 1:-1] = (current_pressure[2:, 1:-1, 1:-1] - current_pressure[:-2, 1:-1, 1:-1]) / (2 * self.dx)
            grad_p_y[1:-1, 1:-1, 1:-1] = (current_pressure[1:-1, 2:, 1:-1] - current_pressure[1:-1, :-2, 1:-1]) / (2 * self.dy)
            grad_p_z[1:-1, 1:-1, 1:-1] = (current_pressure[1:-1, 1:-1, 2:] - current_pressure[1:-1, 1:-1, :-2]) / (2 * self.dz)

            # Boundary handling for pressure gradient (using one-sided differences at boundaries)
            if not self.mesh_info.get('is_periodic_x', False):
                grad_p_x[0,1:-1,1:-1] = (current_pressure[1,1:-1,1:-1] - current_pressure[0,1:-1,1:-1]) / self.dx
                grad_p_x[-1,1:-1,1:-1] = (current_pressure[-1,1:-1,1:-1] - current_pressure[-2,1:-1,1:-1]) / self.dx
            
            if not self.mesh_info.get('is_periodic_y', False):
                grad_p_y[1:-1,0,1:-1] = (current_pressure[1:-1,1,1:-1] - current_pressure[1:-1,0,1:-1]) / self.dy
                grad_p_y[1:-1,-1,1:-1] = (current_pressure[1:-1,-1,1:-1] - current_pressure[1:-1,-2,1:-1]) / self.dy

            if not self.mesh_info.get('is_periodic_z', False):
                grad_p_z[1:-1,1:-1,0] = (current_pressure[1:-1,1:-1,1] - current_pressure[1:-1,1:-1,0]) / self.dz
                grad_p_z[1:-1,1:-1,-1] = (current_pressure[1:-1,1:-1,-1] - current_pressure[1:-1,1:-1,-2]) / self.dz

            print(f"[DEBUG] Grad_p_x stats: min={np.nanmin(grad_p_x):.4e}, max={np.nanmax(grad_p_x):.4e}, has_nan={np.any(np.isnan(grad_p_x))}, has_inf={np.any(np.isinf(grad_p_x))}")
            print(f"[DEBUG] Grad_p_y stats: min={np.nanmin(grad_p_y):.4e}, max={np.nanmax(grad_p_y):.4e}, has_nan={np.any(np.isnan(grad_p_y))}, has_inf={np.any(np.isinf(grad_p_y))}")
            print(f"[DEBUG] Grad_p_z stats: min={np.nanmin(grad_p_z):.4e}, max={np.nanmax(grad_p_z):.4e}, has_nan={np.any(np.isnan(grad_p_z))}, has_inf={np.any(np.isinf(grad_p_z))}")

            # Flatten only the interior part of the pressure gradient
            grad_p_x_flat = grad_p_x[interior_slice_3d].flatten()
            grad_p_y_flat = grad_p_y[interior_slice_3d].flatten()
            grad_p_z_flat = grad_p_z[interior_slice_3d].flatten()

            print(f"[DEBUG ImplicitSolver] u_flat_interior shape: {u_flat_interior.shape}")
            print(f"[DEBUG ImplicitSolver] advection_u_flat shape: {advection_u_flat.shape}")
            print(f"[DEBUG ImplicitSolver] grad_p_x_flat shape: {grad_p_x_flat.shape}")


            # --- Construct RHS for Implicit Diffusion Solver (Momentum Equation) ---
            b_u = u_flat_interior - self.dt * advection_u_flat - (self.dt / self.density) * grad_p_x_flat
            b_v = v_flat_interior - self.dt * advection_v_flat - (self.dt / self.density) * grad_p_y_flat
            b_w = w_flat_interior - self.dt * advection_w_flat - (self.dt / self.density) * grad_p_z_flat

            print(f"[DEBUG] RHS_u stats: min={np.nanmin(b_u):.4e}, max={np.nanmax(b_u):.4e}, has_nan={np.any(np.isnan(b_u))}, has_inf={np.any(np.isinf(b_u))}")
            print(f"[DEBUG] RHS_v stats: min={np.nanmin(b_v):.4e}, max={np.nanmax(b_v):.4e}, has_nan={np.any(np.isnan(b_v))}, has_inf={np.any(np.isinf(b_v))}")
            print(f"[DEBUG] RHS_w stats: min={np.nanmin(b_w):.4e}, max={np.nanmax(b_w):.4e}, has_nan={np.any(np.isinf(b_w))}, has_inf={np.any(np.isinf(b_w))}")


            # --- Solve for Tentative Velocities using Implicit Diffusion Matrix ---
            try:
                u_new_flat = spsolve(self.diffusion_matrix_LHS, b_u)
                v_new_flat = spsolve(self.diffusion_matrix_LHS, b_v)
                w_new_flat = spsolve(self.diffusion_matrix_LHS, b_w)
            except Exception as e:
                print(f"Error solving implicit diffusion system: {e}", file=sys.stderr)
                raise

            # --- Reshape flattened solutions back to 3D interior arrays ---
            u_new_interior = u_new_flat.reshape((nx_interior, ny_interior, nz_interior))
            v_new_interior = v_new_flat.reshape((nx_interior, ny_interior, nz_interior))
            w_new_interior = w_new_flat.reshape((nx_interior, ny_interior, nz_interior))

            # --- Update the interior of current_velocity with the new tentative velocities ---
            current_velocity[interior_slice_3d + (0,)] = u_new_interior
            current_velocity[interior_slice_3d + (1,)] = v_new_interior
            current_velocity[interior_slice_3d + (2,)] = w_new_interior
            
            print(f"[DEBUG] Tentative velocity after implicit solve (before BCs): min={np.nanmin(current_velocity):.4e}, max={np.nanmax(current_velocity):.4e}, has_nan={np.any(np.isnan(current_velocity))}, has_inf={np.any(np.isinf(current_velocity))}")


            # --- Apply Boundary Conditions to the tentative velocity field ---
            current_velocity, _ = apply_boundary_conditions(
                current_velocity,
                current_pressure, # Pressure is passed, but often not used for tentative velocity BCs
                self.fluid_properties_dict,
                self.mesh_info,
                is_tentative_step=True
            )
            print(f"[DEBUG] Tentative velocity AFTER BCs: min={np.nanmin(current_velocity):.4e}, max={np.nanmax(current_velocity):.4e}, has_nan={np.any(np.isnan(current_velocity))}, has_inf={np.any(np.isinf(current_velocity))}")


            # --- Pressure Projection (still explicit in this fractional step) ---
            divergence = compute_pressure_divergence(current_velocity, self.mesh_info)
            print(f"[DEBUG] Divergence stats BEFORE Poisson solve: min={np.nanmin(divergence):.4e}, max={np.nanmax(divergence):.4e}, has_nan={np.any(np.isnan(divergence))}, has_inf={np.any(np.isinf(divergence))}")

            # --- CALLING THE POISSON SOLVER ---
            pressure_correction_phi, poisson_final_residual = solve_poisson_for_phi(
                divergence,
                self.mesh_info,
                self.dt,
                tolerance=poisson_tolerance,
                max_iterations=poisson_max_iter,
                return_residual=True,
                backend="bicgstab",
                preconditioner_type="ilu"
            )
            print(f"[DEBUG] Pressure_correction_phi stats: min={np.nanmin(pressure_correction_phi):.4e}, max={np.nanmax(pressure_correction_phi):.4e}, has_nan={np.any(np.isnan(pressure_correction_phi))}, has_inf={np.any(np.isinf(pressure_correction_phi))}")
            print(f"[DEBUG] Poisson Solver Final Residual: {poisson_final_residual:.6e}")

            if np.any(np.isnan(pressure_correction_phi)) or np.any(np.isinf(pressure_correction_phi)):
                print("WARNING: NaN or Inf detected in pressure_correction_phi. Clamping/Erroring out.", file=sys.stderr)
                max_phi_val = np.finfo(np.float64).max / 10 # Prevent overflow for extremely large numbers
                pressure_correction_phi = np.clip(pressure_correction_phi, -max_phi_val, max_phi_val)


            # Apply pressure correction to velocity and update pressure
            # Unpack all 6 return values from apply_pressure_correction
            current_velocity, current_pressure, _, _, _, divergence_after_correction_field = apply_pressure_correction(
                tentative_velocity_field=current_velocity,
                current_pressure_field=current_pressure,
                phi=pressure_correction_phi,
                dt=self.dt,
                density=self.density,
                mesh_info=self.mesh_info # <--- CORRECTED: Passing the full mesh_info dictionary
            )
            
            print(f"[DEBUG] Velocity AFTER pressure correction (before final BCs): min={np.nanmin(current_velocity):.4e}, max={np.nanmax(current_velocity):.4e}, has_nan={np.any(np.isnan(current_velocity))}, has_inf={np.any(np.isinf(current_velocity))}")
            print(f"[DEBUG] Pressure AFTER pressure correction (before final BCs): min={np.nanmin(current_pressure):.4e}, max={np.nanmax(current_pressure):.4e}, has_nan={np.any(np.isnan(current_pressure))}, has_inf={np.any(np.isinf(current_pressure))}")

            # After correction, apply BCs again to ensure consistency
            current_velocity, current_pressure = apply_boundary_conditions(
                current_velocity,
                current_pressure,
                self.fluid_properties_dict,
                self.mesh_info,
                is_tentative_step=False
            )
            print(f"[DEBUG] Velocity AFTER final BCs for pseudo-iteration: min={np.nanmin(current_velocity):.4e}, max={np.nanmax(current_velocity):.4e}, has_nan={np.any(np.isnan(current_velocity))}, has_inf={np.any(np.isinf(current_velocity))}")
            print(f"[DEBUG] Pressure AFTER final BCs for pseudo-iteration: min={np.nanmin(current_pressure):.4e}, max={np.nanmax(current_pressure):.4e}, has_nan={np.any(np.isnan(current_pressure))}, has_inf={np.any(np.isinf(current_pressure))}")

            # Recalculate divergence after correction and BCs for logging/next pseudo-iteration
            # This line is now redundant as divergence_after_correction_field is returned by apply_pressure_correction
            # divergence_after_correction_field = compute_pressure_divergence(current_velocity, self.mesh_info)
            print(f"[DEBUG] Final divergence for pseudo-iteration: min={np.nanmin(divergence_after_correction_field):.4e}, max={np.nanmax(divergence_after_correction_field):.4e}, has_nan={np.any(np.isnan(divergence_after_correction_field))}, has_inf={np.any(np.isinf(divergence_after_correction_field))}")

        # Final application of BCs after all pseudo-iterations
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            current_velocity,
            current_pressure,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        return updated_velocity_field, updated_pressure_field, divergence_after_correction_field



