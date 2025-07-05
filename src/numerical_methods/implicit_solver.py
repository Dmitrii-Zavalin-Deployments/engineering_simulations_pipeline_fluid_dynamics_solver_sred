# src/numerical_methods/implicit_solver.py (Updated with correct __init__ and debug prints)

import numpy as np
import sys
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import spsolve

# Import the individual numerical methods and the boundary conditions module.
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    from .pressure_correction import apply_pressure_correction
    from physics.boundary_conditions_applicator import apply_boundary_conditions
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
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
        Initializes the ImplicitSolver.

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
        self.dt = dt

        self.nx_int, self.ny_int, self.nz_int = self.mesh_info['grid_shape']
        self.dx = self.mesh_info['dx']
        self.dy = self.mesh_info['dy']
        self.dz = self.mesh_info['dz']

        # Pre-build the LHS matrix for the implicit diffusion step
        # This matrix is constant as long as dt and mesh_info don't change
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
        N = self.nx_int * self.ny_int * self.nz_int
        LHS_matrix = lil_matrix((N, N))

        # Identity matrix part
        LHS_matrix.setdiag(1.0)

        # Laplacian part
        nu_dt = self.viscosity * self.dt

        # Coefficients for finite difference Laplacian (negative for diffusion operator)
        coeff_x = nu_dt / (self.dx ** 2)
        coeff_y = nu_dt / (self.dy ** 2)
        coeff_z = nu_dt / (self.dz ** 2)

        # Iterate over all interior cells
        for k in range(self.nz_int):
            for j in range(self.ny_int):
                for i in range(self.nx_int):
                    idx = i + j * self.nx_int + k * self.nx_int * self.ny_int

                    # Self-term for Laplacian (sum of coefficients, times -1 for -Laplacian)
                    # This adds to the diagonal, effectively making it (1 + 2*coeff_x + 2*coeff_y + 2*coeff_z)
                    LHS_matrix[idx, idx] += 2 * coeff_x + 2 * coeff_y + 2 * coeff_z

                    # Neighbors (subtracting from diagonal implies adding to neighbors on RHS)
                    # x-direction
                    if i > 0:
                        LHS_matrix[idx, idx - 1] = -coeff_x
                    if i < self.nx_int - 1:
                        LHS_matrix[idx, idx + 1] = -coeff_x

                    # y-direction
                    if j > 0:
                        LHS_matrix[idx, idx - self.nx_int] = -coeff_y
                    if j < self.ny_int - 1:
                        LHS_matrix[idx, idx + self.nx_int] = -coeff_y

                    # z-direction
                    if k > 0:
                        LHS_matrix[idx, idx - self.nx_int * self.ny_int] = -coeff_z
                    if k < self.nz_int - 1:
                        LHS_matrix[idx, idx + self.nx_int * self.ny_int] = -coeff_z

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
                                         Shape: (nx+2, ny+2, nz+2, 3).
            pressure_field (np.ndarray): Current pressure field. Shape: (nx+2, ny+2, nz+2).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Updated velocity_field, pressure_field, and the divergence field after correction.
        """
        
        current_velocity = np.copy(velocity_field)
        current_pressure = np.copy(pressure_field)
        
        num_pseudo_iterations = 2 # Or 1, depending on desired accuracy vs. performance
        poisson_tolerance = 1e-6
        poisson_max_iter = 1000

        nx_int, ny_int, nz_int = self.mesh_info['grid_shape']
        
        divergence_after_correction_field = np.zeros_like(current_pressure)

        print(f"[DEBUG] Initial velocity in step(): min={np.nanmin(current_velocity):.4e}, max={np.nanmax(current_velocity):.4e}, has_nan={np.any(np.isnan(current_velocity))}, has_inf={np.any(np.isinf(current_velocity))}")
        print(f"[DEBUG] Initial pressure in step(): min={np.nanmin(current_pressure):.4e}, max={np.nanmax(current_pressure):.4e}, has_nan={np.any(np.isnan(current_pressure))}, has_inf={np.any(np.isinf(current_pressure))}")


        for pseudo_iter in range(num_pseudo_iterations):
            # Reshape velocity components for linear solver (flattened 1D arrays for interior)
            u_flat = current_velocity[1:-1, 1:-1, 1:-1, 0].flatten()
            v_flat = current_velocity[1:-1, 1:-1, 1:-1, 1].flatten()
            w_flat = current_velocity[1:-1, 1:-1, 1:-1, 2].flatten()

            # --- Explicit Advection Term ---
            advection_u = compute_advection_term(current_velocity[..., 0], current_velocity, self.mesh_info)
            advection_v = compute_advection_term(current_velocity[..., 1], current_velocity, self.mesh_info)
            advection_w = compute_advection_term(current_velocity[..., 2], current_velocity, self.mesh_info)
            
            print(f"[DEBUG] Advection_u stats: min={np.nanmin(advection_u):.4e}, max={np.nanmax(advection_u):.4e}, has_nan={np.any(np.isnan(advection_u))}, has_inf={np.any(np.isinf(advection_u))}")
            print(f"[DEBUG] Advection_v stats: min={np.nanmin(advection_v):.4e}, max={np.nanmax(advection_v):.4e}, has_nan={np.any(np.isnan(advection_v))}, has_inf={np.any(np.isinf(advection_v))}")
            print(f"[DEBUG] Advection_w stats: min={np.nanmin(advection_w):.4e}, max={np.nanmax(advection_w):.4e}, has_nan={np.any(np.isnan(advection_w))}, has_inf={np.any(np.isinf(advection_w))}")


            advection_u_flat = advection_u[1:-1, 1:-1, 1:-1].flatten()
            advection_v_flat = advection_v[1:-1, 1:-1, 1:-1].flatten()
            advection_w_flat = advection_w[1:-1, 1:-1, 1:-1].flatten()

            # --- Pressure Gradient Term (Explicit) ---
            grad_p_x = np.zeros_like(current_pressure)
            grad_p_y = np.zeros_like(current_pressure)
            grad_p_z = np.zeros_like(current_pressure)

            # For internal cells (standard central difference)
            grad_p_x[1:-1, :, :] = (current_pressure[2:, :, :] - current_pressure[:-2, :, :]) / (2 * self.mesh_info['dx'])
            grad_p_y[:, 1:-1, :] = (current_pressure[:, 2:, :] - current_pressure[:, :-2, :]) / (2 * self.mesh_info['dy'])
            grad_p_z[:, :, 1:-1] = (current_pressure[:, :, 2:] - current_pressure[:, :, :-2]) / (2 * self.mesh_info['dz'])

            # Boundary handling for pressure gradient
            if not self.mesh_info.get('is_periodic_x', False):
                grad_p_x[0,:,:] = (current_pressure[1,:,:] - current_pressure[0,:,:]) / self.mesh_info['dx']
                grad_p_x[-1,:,:] = (current_pressure[-1,:,:] - current_pressure[-2,:,:]) / self.mesh_info['dx']
            
            if not self.mesh_info.get('is_periodic_y', False):
                grad_p_y[:,0,:] = (current_pressure[:,1,:] - current_pressure[:,0,:]) / self.mesh_info['dy']
                grad_p_y[:,-1,:] = (current_pressure[:,-1,:] - current_pressure[:,-2,:]) / self.mesh_info['dy']

            if not self.mesh_info.get('is_periodic_z', False):
                grad_p_z[:,:,0] = (current_pressure[:,:,1] - current_pressure[:,:,0]) / self.mesh_info['dz']
                grad_p_z[:,:,-1] = (current_pressure[:,:,-1] - current_pressure[:,:,-2]) / self.mesh_info['dz']

            print(f"[DEBUG] Grad_p_x stats: min={np.nanmin(grad_p_x):.4e}, max={np.nanmax(grad_p_x):.4e}, has_nan={np.any(np.isnan(grad_p_x))}, has_inf={np.any(np.isinf(grad_p_x))}")
            print(f"[DEBUG] Grad_p_y stats: min={np.nanmin(grad_p_y):.4e}, max={np.nanmax(grad_p_y):.4e}, has_nan={np.any(np.isnan(grad_p_y))}, has_inf={np.any(np.isinf(grad_p_y))}")
            print(f"[DEBUG] Grad_p_z stats: min={np.nanmin(grad_p_z):.4e}, max={np.nanmax(grad_p_z):.4e}, has_nan={np.any(np.isnan(grad_p_z))}, has_inf={np.any(np.isinf(grad_p_z))}")

            grad_p_x_flat = grad_p_x[1:-1, 1:-1, 1:-1].flatten()
            grad_p_y_flat = grad_p_y[1:-1, 1:-1, 1:-1].flatten()
            grad_p_z_flat = grad_p_z[1:-1, 1:-1, 1:-1].flatten()


            # --- Construct RHS for Implicit Diffusion Solver (Momentum Equation) ---
            b_u = u_flat - self.dt * advection_u_flat - (self.dt / self.density) * grad_p_x_flat
            b_v = v_flat - self.dt * advection_v_flat - (self.dt / self.density) * grad_p_y_flat
            b_w = w_flat - self.dt * advection_w_flat - (self.dt / self.density) * grad_p_z_flat

            print(f"[DEBUG] RHS_u stats: min={np.nanmin(b_u):.4e}, max={np.nanmax(b_u):.4e}, has_nan={np.any(np.isnan(b_u))}, has_inf={np.any(np.isinf(b_u))}")
            print(f"[DEBUG] RHS_v stats: min={np.nanmin(b_v):.4e}, max={np.nanmax(b_v):.4e}, has_nan={np.any(np.isnan(b_v))}, has_inf={np.any(np.isinf(b_v))}")
            print(f"[DEBUG] RHS_w stats: min={np.nanmin(b_w):.4e}, max={np.nanmax(b_w):.4e}, has_nan={np.any(np.isnan(b_w))}, has_inf={np.any(np.isinf(b_w))}")


            # --- Solve for Tentative Velocities using Implicit Diffusion Matrix ---
            try:
                u_new_flat = spsolve(self.diffusion_matrix_LHS, b_u)
                v_new_flat = spsolve(self.diffusion_matrix_LHS, b_v)
                w_new_flat = spsolve(self.diffusion_matrix_LHS, b_w)
            except Exception as e:
                print(f"Error solving implicit diffusion system: {e}", file=sys.stderr)
                raise

            u_temp_full = np.zeros_like(current_velocity[..., 0])
            v_temp_full = np.zeros_like(current_velocity[..., 1])
            w_temp_full = np.zeros_like(current_velocity[..., 2])

            u_temp_full[1:-1, 1:-1, 1:-1] = u_new_flat.reshape((nx_int, ny_int, nz_int))
            v_temp_full[1:-1, 1:-1, 1:-1] = v_new_flat.reshape((nx_int, ny_int, nz_int))
            w_temp_full[1:-1, 1:-1, 1:-1] = w_new_flat.reshape((nx_int, ny_int, nz_int))

            current_velocity[..., 0] = u_temp_full
            current_velocity[..., 1] = v_temp_full
            current_velocity[..., 2] = w_temp_full
            
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

            pressure_correction_phi = solve_poisson_for_phi(
                divergence,
                self.mesh_info,
                self.dt,
                tolerance=poisson_tolerance,
                max_iterations=poisson_max_iter
            )
            
            print(f"[DEBUG] Pressure_correction_phi stats: min={np.nanmin(pressure_correction_phi):.4e}, max={np.nanmax(pressure_correction_phi):.4e}, has_nan={np.any(np.isnan(pressure_correction_phi))}, has_inf={np.any(np.isinf(pressure_correction_phi))}")
            if np.any(np.isnan(pressure_correction_phi)) or np.any(np.isinf(pressure_correction_phi)):
                print("WARNING: NaN or Inf detected in pressure_correction_phi. Clamping/Erroring out.", file=sys.stderr)
                max_phi_val = np.finfo(np.float64).max / 10 # Prevent overflow for extremely large numbers
                pressure_correction_phi = np.clip(pressure_correction_phi, -max_phi_val, max_phi_val)


            # Apply pressure correction to velocity and update pressure
            current_velocity, current_pressure = apply_pressure_correction(
                current_velocity,
                current_pressure,
                pressure_correction_phi,
                self.mesh_info,
                self.dt,
                self.density
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

            # Recalculate divergence after correction and BCs
            divergence_after_correction_field = compute_pressure_divergence(current_velocity, self.mesh_info)
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


