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
        # Correctly extract nx, ny, nz from mesh_info['grid_shape']
        # Also ensure dx, dy, dz are directly from mesh_info
        nx_grid, ny_grid, nz_grid = self.mesh_info['grid_shape'] # Get nx, ny, nz from grid_shape
        self.diffusion_matrix_LHS = self._build_diffusion_matrix(
            nx_grid, ny_grid, nz_grid,
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
                    # Assuming mesh_info['is_periodic_x'] is defined
                    elif self.mesh_info.get('is_periodic_x', False):
                        A[idx, idx + (nx - 1)] = -coeff_x # Wrap around
                    
                    if i < nx - 1:
                        A[idx, idx + 1] = -coeff_x
                    elif self.mesh_info.get('is_periodic_x', False):
                        A[idx, idx - (nx - 1)] = -coeff_x # Wrap around

                    # Y-direction
                    if j > 0:
                        A[idx, idx - nx] = -coeff_y
                    elif self.mesh_info.get('is_periodic_y', False):
                        A[idx, idx + (ny - 1) * nx] = -coeff_y # Wrap around
                    
                    if j < ny - 1:
                        A[idx, idx + nx] = -coeff_y
                    elif self.mesh_info.get('is_periodic_y', False):
                        A[idx, idx - (ny - 1) * nx] = -coeff_y # Wrap around

                    # Z-direction
                    if k > 0:
                        A[idx, idx - nx * ny] = -coeff_z
                    elif self.mesh_info.get('is_periodic_z', False):
                        A[idx, idx + (nz - 1) * nx * ny] = -coeff_z # Wrap around
                    
                    if k < nz - 1:
                        A[idx, idx + nx * ny] = -coeff_z
                    elif self.mesh_info.get('is_periodic_z', False):
                        A[idx, idx - (nz - 1) * nx * ny] = -coeff_z # Wrap around
        
        # Convert to CSR format for efficient solving
        return A.tocsr()

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs one semi-implicit time step, using implicit diffusion.

        Args:
            velocity_field (np.ndarray): Current velocity field (U, V, W components).
                                         Shape: (nx, ny, nz, 3).
            pressure_field (np.ndarray): Current pressure field. Shape: (nx, ny, nz).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Updated velocity_field, pressure_field, and the divergence field after correction.
        """
        
        current_velocity = np.copy(velocity_field)
        current_pressure = np.copy(pressure_field)
        
        # Parameters for the pseudo-iterations (PISO-like structure)
        num_pseudo_iterations = 2
        poisson_tolerance = 1e-6
        poisson_max_iter = 1000

        # Get the actual grid dimensions for interior cells from mesh_info['grid_shape']
        # This assumes velocity_field and pressure_field have ghost cells, so actual interior size is (nx,ny,nz)
        nx_int, ny_int, nz_int = self.mesh_info['grid_shape']
        
        # Reshape velocity components for linear solver (flattened 1D arrays)
        # Only flatten the INTERIOR cells for the solver, not the ghost cells
        u_flat = current_velocity[1:-1, 1:-1, 1:-1, 0].flatten()
        v_flat = current_velocity[1:-1, 1:-1, 1:-1, 1].flatten()
        w_flat = current_velocity[1:-1, 1:-1, 1:-1, 2].flatten()


        # Initialize divergence_after_correction_field for return
        # Its shape should match the pressure field, including ghost cells.
        divergence_after_correction_field = np.zeros_like(current_pressure)

        for pseudo_iter in range(num_pseudo_iterations):
            # --- Explicit Advection Term ---
            advection_u = compute_advection_term(current_velocity[..., 0], current_velocity, self.mesh_info)
            advection_v = compute_advection_term(current_velocity[..., 1], current_velocity, self.mesh_info)
            advection_w = compute_advection_term(current_velocity[..., 2], current_velocity, self.mesh_info)

            # Reshape advection terms to 1D, considering only interior cells
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
            # This logic should typically be handled by boundary conditions applicator or be more robust
            # For simplicity for now, using forward/backward diff at boundaries.
            # However, for pressure correction, this gradient needs to be consistent with how apply_pressure_correction
            # uses the pressure_correction_phi gradient.
            # A more robust solution might involve passing ghost cell pressure values to the pressure gradient calculation.
            if not self.mesh_info.get('is_periodic_x', False):
                grad_p_x[0,:,:] = (current_pressure[1,:,:] - current_pressure[0,:,:]) / self.mesh_info['dx'] # Forward diff
                grad_p_x[-1,:,:] = (current_pressure[-1,:,:] - current_pressure[-2,:,:]) / self.mesh_info['dx'] # Backward diff
            if not self.mesh_info.get('is_periodic_y', False):
                grad_p_y[:,0,:] = (current_pressure[:,1,:] - current_pressure[:,0,:]) / self.mesh_info['dy']
                grad_p_y[:,-1,:] = (current_pressure[:,-1,:] - current_pressure[:,-2,:]) / self.mesh_info['dy']
            if not self.mesh_info.get('is_periodic_z', False):
                grad_p_z[:,:,0] = (current_pressure[:,:,1] - current_pressure[:,:,0]) / self.mesh_info['dz']
                grad_p_z[:,:,-1] = (current_pressure[:,:,-1] - current_pressure[:,-2,:]) / self.mesh_info['dz']

            # Extract interior cells for flattening
            grad_p_x_flat = grad_p_x[1:-1, 1:-1, 1:-1].flatten()
            grad_p_y_flat = grad_p_y[1:-1, 1:-1, 1:-1].flatten()
            grad_p_z_flat = grad_p_z[1:-1, 1:-1, 1:-1].flatten()


            # --- Construct RHS for Implicit Diffusion Solver (Momentum Equation) ---
            b_u = u_flat - self.dt * advection_u_flat - (self.dt / self.density) * grad_p_x_flat
            b_v = v_flat - self.dt * advection_v_flat - (self.dt / self.density) * grad_p_y_flat
            b_w = w_flat - self.dt * advection_w_flat - (self.dt / self.density) * grad_p_z_flat

            # --- Solve for Tentative Velocities using Implicit Diffusion Matrix ---
            try:
                u_new_flat = spsolve(self.diffusion_matrix_LHS, b_u)
                v_new_flat = spsolve(self.diffusion_matrix_LHS, b_v)
                w_new_flat = spsolve(self.diffusion_matrix_LHS, b_w)
            except Exception as e:
                print(f"Error solving implicit diffusion system: {e}", file=sys.stderr)
                raise

            # Reshape solved velocities back into an array with ghost cells
            # Create temporary full arrays with ghost cells for current_velocity update
            u_temp_full = np.zeros_like(current_velocity[..., 0])
            v_temp_full = np.zeros_like(current_velocity[..., 1])
            w_temp_full = np.zeros_like(current_velocity[..., 2])

            u_temp_full[1:-1, 1:-1, 1:-1] = u_new_flat.reshape((nx_int, ny_int, nz_int))
            v_temp_full[1:-1, 1:-1, 1:-1] = v_new_flat.reshape((nx_int, ny_int, nz_int))
            w_temp_full[1:-1, 1:-1, 1:-1] = w_new_flat.reshape((nx_int, ny_int, nz_int))

            current_velocity[..., 0] = u_temp_full
            current_velocity[..., 1] = v_temp_full
            current_velocity[..., 2] = w_temp_full


            # --- Apply Boundary Conditions to the tentative velocity field ---
            current_velocity, _ = apply_boundary_conditions(
                current_velocity,
                current_pressure,
                self.fluid_properties_dict,
                self.mesh_info,
                is_tentative_step=True
            )

            # --- Pressure Projection (still explicit in this fractional step) ---
            divergence = compute_pressure_divergence(current_velocity, self.mesh_info)

            pressure_correction_phi = solve_poisson_for_phi(
                divergence,
                self.mesh_info,
                self.dt,
                tolerance=poisson_tolerance,
                max_iterations=poisson_max_iter
            )
            
            if np.any(np.isnan(pressure_correction_phi)) or np.any(np.isinf(pressure_correction_phi)):
                print("WARNING: NaN or Inf detected in pressure_correction_phi. Clamping/Erroring out.", file=sys.stderr)
                max_phi_val = np.finfo(np.float64).max / 10
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
                is_tentative_step=False
            )
            
            # Recalculate divergence after correction and BCs
            divergence_after_correction_field = compute_pressure_divergence(current_velocity, self.mesh_info)

        # Final application of BCs after all pseudo-iterations
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            current_velocity,
            current_pressure,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        return updated_velocity_field, updated_pressure_field, divergence_after_correction_field



