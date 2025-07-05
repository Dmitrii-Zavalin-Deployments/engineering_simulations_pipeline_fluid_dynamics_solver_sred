# src/numerical_methods/implicit_solver.py

import numpy as np
import sys
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import spsolve

# Import the individual numerical methods and the boundary conditions module.
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term # This will still be used to calculate RHS term
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
    An improved semi-implicit solver for the Navier-Stokes equations,
    treating the diffusion term implicitly for enhanced stability.

    This class sets up and solves a linear system for the diffusion term
    at each time step using SciPy's sparse linear algebra capabilities.
    Advection and pressure projection are handled in a fractional step manner.
    """

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        """
        Initializes the implicit solver with simulation parameters and
        pre-computes the implicit diffusion matrix.

        Args:
            fluid_properties (dict): Dictionary with 'density' and 'viscosity'.
            mesh_info (dict): Dictionary containing structured grid information and
                              pre-processed boundary condition data.
            dt (float): Time step size.
        """
        self.density = fluid_properties['density']
        self.viscosity = fluid_properties['viscosity']
        self.dt = dt
        self.mesh_info = mesh_info

        # This dictionary is created once in the constructor for BC application
        self.fluid_properties_dict = fluid_properties

        # Pre-compute the matrix for the implicit diffusion step
        self.diffusion_matrix_LHS = self._build_diffusion_matrix(
            self.mesh_info['nx'], self.mesh_info['ny'], self.mesh_info['nz'],
            self.mesh_info['dx'], self.mesh_info['dy'], self.mesh_info['dz']
        )
        print("Implicit diffusion matrix pre-computed.")


    def _build_diffusion_matrix(self, nx: int, ny: int, nz: int, dx: float, dy: float, dz: float):
        """
        Builds the sparse matrix for the implicit diffusion term (I - dt * nu/rho * Laplacian).
        Uses a 7-point stencil for the 3D Laplacian.

        Args:
            nx, ny, nz (int): Number of grid cells in each dimension.
            dx, dy, dz (float): Grid spacing in each dimension.

        Returns:
            scipy.sparse.csr_matrix: The assembled sparse matrix for the implicit diffusion.
        """
        total_cells = nx * ny * nz
        
        # Identity matrix for the (I * u_new) part
        A = identity(total_cells, format='lil') 

        nu_over_rho_dt = (self.viscosity / self.density) * self.dt

        # Coefficients for the Laplacian operator
        coeff_x = nu_over_rho_dt / (dx * dx)
        coeff_y = nu_over_rho_dt / (dy * dy)
        coeff_z = nu_over_rho_dt / (dz * dz)

        # Iterate over all cells to build the matrix A
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    idx = i + j * nx + k * nx * ny # 1D index for the current cell

                    # Diagonal element: 1 + dt * nu/rho * (2/dx^2 + 2/dy^2 + 2/dz^2)
                    A[idx, idx] = 1.0 + 2 * coeff_x + 2 * coeff_y + 2 * coeff_z

                    # Neighbors (implicit contribution)
                    # X-direction
                    if i > 0:
                        A[idx, idx - 1] = -coeff_x
                    elif self.mesh_info['is_periodic_x']:
                        A[idx, idx + (nx - 1)] = -coeff_x # Wrap around
                    
                    if i < nx - 1:
                        A[idx, idx + 1] = -coeff_x
                    elif self.mesh_info['is_periodic_x']:
                        A[idx, idx - (nx - 1)] = -coeff_x # Wrap around

                    # Y-direction
                    if j > 0:
                        A[idx, idx - nx] = -coeff_y
                    elif self.mesh_info['is_periodic_y']:
                        A[idx, idx + (ny - 1) * nx] = -coeff_y # Wrap around
                    
                    if j < ny - 1:
                        A[idx, idx + nx] = -coeff_y
                    elif self.mesh_info['is_periodic_y']:
                        A[idx, idx - (ny - 1) * nx] = -coeff_y # Wrap around

                    # Z-direction
                    if k > 0:
                        A[idx, idx - nx * ny] = -coeff_z
                    elif self.mesh_info['is_periodic_z']:
                        A[idx, idx + (nz - 1) * nx * ny] = -coeff_z # Wrap around
                    
                    if k < nz - 1:
                        A[idx, idx + nx * ny] = -coeff_z
                    elif self.mesh_info['is_periodic_z']:
                        A[idx, idx - (nz - 1) * nx * ny] = -coeff_z # Wrap around
        
        # Convert to CSR format for efficient solving
        return A.tocsr()

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs one semi-implicit time step, using implicit diffusion.

        Args:
            velocity_field (np.ndarray): Current velocity field (U, V, W components).
                                         Shape: (nx, ny, nz, 3).
            pressure_field (np.ndarray): Current pressure field. Shape: (nx, ny, nz).

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated velocity_field and pressure_field after one time step.
        """

        print(f"Running semi-implicit solver step (dt={self.dt}).")
        
        current_velocity = np.copy(velocity_field)
        current_pressure = np.copy(pressure_field)
        
        # Parameters for the pseudo-iterations (PISO-like structure)
        num_pseudo_iterations = 2 # A reduced number might be fine with implicit diffusion.
                                  # You might need to experiment with this value.
        poisson_tolerance = 1e-6
        poisson_max_iter = 1000

        nx, ny, nz, _ = current_velocity.shape
        total_cells = nx * ny * nz

        # Reshape velocity components for linear solver (flattened 1D arrays)
        u_flat = current_velocity[..., 0].flatten()
        v_flat = current_velocity[..., 1].flatten()
        w_flat = current_velocity[..., 2].flatten() # Even if 2D, this is fine, nz=1

        for pseudo_iter in range(num_pseudo_iterations):
            print(f"  Pseudo-iteration {pseudo_iter + 1}/{num_pseudo_iterations}")

            # --- Explicit Advection Term ---
            # Advection is still explicit, calculated from the current 'tentative' velocity.
            advection_u = compute_advection_term(current_velocity[..., 0], current_velocity, self.mesh_info)
            advection_v = compute_advection_term(current_velocity[..., 1], current_velocity, self.mesh_info)
            advection_w = compute_advection_term(current_velocity[..., 2], current_velocity, self.mesh_info)

            # Reshape advection terms to 1D
            advection_u_flat = advection_u.flatten()
            advection_v_flat = advection_v.flatten()
            advection_w_flat = advection_w.flatten()

            # --- Pressure Gradient Term (Explicit) ---
            # Compute pressure gradient using central differences, similar to pressure_correction's philosophy
            # We need the pressure gradient for the momentum equation.
            # This is an explicit term, using current_pressure.
            grad_p_x = np.zeros_like(current_pressure)
            grad_p_y = np.zeros_like(current_pressure)
            grad_p_z = np.zeros_like(current_pressure)

            # For internal cells (standard central difference)
            grad_p_x[1:-1, :, :] = (current_pressure[2:, :, :] - current_pressure[:-2, :, :]) / (2 * self.mesh_info['dx'])
            grad_p_y[:, 1:-1, :] = (current_pressure[:, 2:, :] - current_pressure[:, :-2, :]) / (2 * self.mesh_info['dy'])
            grad_p_z[:, :, 1:-1] = (current_pressure[:, :, 2:] - current_pressure[:, :, :-2]) / (2 * self.mesh_info['dz'])

            # Boundary handling for pressure gradient (e.g., for Neumann walls, dp/dn=0)
            # This part should be consistent with how pressure boundary conditions are handled in pressure_divergence
            # For simplicity here, we're extending the gradient from inside.
            # A more robust way would involve specific boundary stencils or ghost cells.
            # For Dirichlet boundaries (pressure specified), gradient might be known.
            # For Neumann boundaries (velocity normal specified), gradient at boundary can be zero or inferred.
            
            # Simple boundary approximation for gradient (forward/backward difference at edges)
            # Note: This is a simplified boundary treatment for pressure gradient.
            # For inlet/outlet pressure BCs, the gradient at the boundary would be influenced by that BC.
            # For solid wall (no-slip), pressure gradient normal to wall is often zero.
            if not self.mesh_info['is_periodic_x']:
                grad_p_x[0,:,:] = (current_pressure[1,:,:] - current_pressure[0,:,:]) / self.mesh_info['dx'] # Forward diff
                grad_p_x[-1,:,:] = (current_pressure[-1,:,:] - current_pressure[-2,:,:]) / self.mesh_info['dx'] # Backward diff
            if not self.mesh_info['is_periodic_y']:
                grad_p_y[:,0,:] = (current_pressure[:,1,:] - current_pressure[:,0,:]) / self.mesh_info['dy']
                grad_p_y[:,-1,:] = (current_pressure[:,-1,:] - current_pressure[:,-2,:]) / self.mesh_info['dy']
            if not self.mesh_info['is_periodic_z']:
                grad_p_z[:,:,0] = (current_pressure[:,:,1] - current_pressure[:,:,0]) / self.mesh_info['dz']
                grad_p_z[:,:,-1] = (current_pressure[:,:,-1] - current_pressure[:,:,-2]) / self.mesh_info['dz']

            grad_p_x_flat = grad_p_x.flatten()
            grad_p_y_flat = grad_p_y.flatten()
            grad_p_z_flat = grad_p_z.flatten()


            # --- Construct RHS for Implicit Diffusion Solver (Momentum Equation) ---
            # The momentum equation is:
            # dU/dt = - (U.grad)U + nu * Laplacian(U) - (1/rho) * grad(P)
            # Rearranging for Implicit Diffusion (Backward Euler in time for Diffusion, Explicit for others):
            # (I - dt * nu/rho * Laplacian) U_new = U_old - dt * (U.grad)U_old - dt * (1/rho) * grad(P)_old
            # So, RHS = U_old - dt * Advection_old - dt * (1/rho) * grad(P)_old

            # U-momentum RHS
            b_u = u_flat - self.dt * advection_u_flat - (self.dt / self.density) * grad_p_x_flat
            
            # V-momentum RHS
            b_v = v_flat - self.dt * advection_v_flat - (self.dt / self.density) * grad_p_y_flat
            
            # W-momentum RHS
            b_w = w_flat - self.dt * advection_w_flat - (self.dt / self.density) * grad_p_z_flat

            # --- Solve for Tentative Velocities using Implicit Diffusion Matrix ---
            try:
                u_new_flat = spsolve(self.diffusion_matrix_LHS, b_u)
                v_new_flat = spsolve(self.diffusion_matrix_LHS, b_v)
                w_new_flat = spsolve(self.diffusion_matrix_LHS, b_w)
            except Exception as e:
                print(f"Error solving implicit diffusion system: {e}", file=sys.stderr)
                # Fallback to current velocity or raise error depending on robustness needs
                # For now, let's raise to stop early and investigate
                raise

            # Reshape solved velocities back to 3D grid
            current_velocity[..., 0] = u_new_flat.reshape((nx, ny, nz))
            current_velocity[..., 1] = v_new_flat.reshape((nx, ny, nz))
            current_velocity[..., 2] = w_new_flat.reshape((nx, ny, nz))

            # --- Apply Boundary Conditions to the tentative velocity field ---
            # BCs need to be applied after each momentum prediction step to maintain
            # consistency.
            current_velocity, _ = apply_boundary_conditions(
                current_velocity,
                current_pressure, # Pressure is not modified by tentative step BCs, so its current value is fine
                self.fluid_properties_dict,
                self.mesh_info,
                is_tentative_step=True
            )

            # --- Pressure Projection (still explicit in this fractional step) ---
            # Calculate divergence of the *tentative* velocity field
            divergence = compute_pressure_divergence(current_velocity, self.mesh_info)
            max_divergence_before_correction = np.max(np.abs(divergence))
            print(f"    Max divergence before correction: {max_divergence_before_correction:.2e}")

            # Solve Poisson equation for pressure correction (phi)
            pressure_correction_phi = solve_poisson_for_phi(
                divergence,
                self.mesh_info,
                self.dt,
                tolerance=poisson_tolerance,
                max_iterations=poisson_max_iter
            )
            
            # Check for NaN/Inf in phi
            if np.any(np.isnan(pressure_correction_phi)) or np.any(np.isinf(pressure_correction_phi)):
                print("WARNING: NaN or Inf detected in pressure_correction_phi. Clamping/Erroring out.", file=sys.stderr)
                # You might want to clamp or raise an error here if this happens frequently
                # For now, let's try to clamp to avoid complete explosion, though it indicates an underlying issue.
                max_phi_val = np.finfo(np.float64).max / 10 # A large but not infinite number
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
            
            # After correction, apply BCs again to ensure consistency
            current_velocity, current_pressure = apply_boundary_conditions(
                current_velocity,
                current_pressure,
                self.fluid_properties_dict,
                self.mesh_info,
                is_tentative_step=False # This is the "final" step for this pseudo-iteration
            )
            
            # Recalculate divergence after correction and BCs
            divergence_after_correction = compute_pressure_divergence(current_velocity, self.mesh_info)
            max_divergence_after_correction = np.max(np.abs(divergence_after_correction))
            print(f"    Max divergence after correction: {max_divergence_after_correction:.2e}")

        # Final application of BCs after all pseudo-iterations
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            current_velocity,
            current_pressure,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        return updated_velocity_field, updated_pressure_field



