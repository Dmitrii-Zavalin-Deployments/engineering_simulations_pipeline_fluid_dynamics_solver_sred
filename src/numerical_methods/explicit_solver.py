# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

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

class ExplicitSolver:
    """
    An explicit solver for the incompressible Navier-Stokes equations
    using a fractional step (Projection) method.
    """

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        """
        Initializes the explicit solver.

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

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: # Keep the return type consistent
        """
        Performs one explicit time step using the fractional step method.

        Args:
            velocity_field (np.ndarray): Current velocity field (U, V, W components).
                                         Shape: (nx+2, ny+2, nz+2, 3).
            pressure_field (np.ndarray): Current pressure field. Shape: (nx+2, ny+2, nz+2).
            # Removed step_count and current_time

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Updated velocity_field, pressure_field, and the divergence field after correction.
        """
        
        # print(f"Running explicit solver step (dt={self.dt}).") # Commented out, log_flow_metrics will handle progress printing

        # 1. Apply boundary conditions to current fields
        # This ensures that any initial conditions or previous step's results
        # are consistent with BCs before computing derivatives.
        velocity_field, pressure_field = apply_boundary_conditions(
            velocity_field,
            pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False # Initial application
        )

        # 2. Compute advection and diffusion terms using current velocity
        # These terms use the values at 'n' to compute u_star.
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        diffusion_u = compute_diffusion_term(velocity_field[..., 0], self.viscosity, self.density, self.mesh_info)
        diffusion_v = compute_diffusion_term(velocity_field[..., 1], self.viscosity, self.density, self.mesh_info)
        diffusion_w = compute_diffusion_term(velocity_field[..., 2], self.viscosity, self.density, self.mesh_info)

        # 3. Predict an intermediate (tentative) velocity field (u_star)
        # This step does NOT include the pressure gradient term.
        # u_star = u_n + dt * [-(u_n . grad)u_n + nu * Laplacian(u_n)]
        
        # Initialize u_star, v_star, w_star with current velocity values (u_n)
        u_star = np.copy(velocity_field[..., 0])
        v_star = np.copy(velocity_field[..., 1])
        w_star = np.copy(velocity_field[..., 2])

        # Update u_star, v_star, w_star with advection and diffusion terms
        u_star += self.dt * (-advection_u + diffusion_u)
        v_star += self.dt * (-advection_v + diffusion_v)
        w_star += self.dt * (-advection_w + diffusion_w)
        
        # Stack u_star, v_star, w_star into a tentative_velocity_field
        tentative_velocity_field = np.stack((u_star, v_star, w_star), axis=-1)

        # 4. Apply boundary conditions to the tentative velocity field
        # For tentative velocity, typically only kinematic (velocity) BCs are applied.
        # Pressure BCs are typically applied during the pressure solve/correction.
        tentative_velocity_field, _ = apply_boundary_conditions(
            tentative_velocity_field,
            pressure_field, # Pressure not modified, just passed for signature compatibility
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=True
        )

        # 5. Solve for pressure correction (Poisson equation)
        # Calculate divergence of the tentative velocity field
        divergence = compute_pressure_divergence(tentative_velocity_field, self.mesh_info)

        # Solve Poisson equation for phi (pressure correction potential)
        # Laplacian(phi) = (1/dt) * divergence(u_star)
        # phi = dt * pressure_correction / rho
        pressure_correction_phi = solve_poisson_for_phi(
            divergence,
            self.mesh_info,
            self.dt,
            tolerance=1e-6, # Example tolerance
            max_iterations=1000 # Example max iterations
        )
        
        # Check for NaN/Inf in phi
        if np.any(np.isnan(pressure_correction_phi)) or np.any(np.isinf(pressure_correction_phi)):
            print("WARNING: NaN or Inf detected in pressure_correction_phi. Clamping/Erroring out.", file=sys.stderr)
            max_phi_val = np.finfo(np.float64).max / 10 # A large but not infinite number
            pressure_correction_phi = np.clip(pressure_correction_phi, -max_phi_val, max_phi_val)


        # 6. Correct the tentative velocity field and update pressure
        # u_n+1 = u_star - dt * (1/rho) * grad(p_correction)
        # p_n+1 = p_n + p_correction
        updated_velocity_field, updated_pressure_field = apply_pressure_correction(
            tentative_velocity_field,
            pressure_field,
            pressure_correction_phi,
            self.mesh_info,
            self.dt,
            self.density
        )

        # 7. Apply final boundary conditions to the corrected fields
        # This step ensures that the final velocity and pressure fields
        # strictly adhere to all boundary conditions for the next time step.
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            updated_velocity_field,
            updated_pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        # Calculate final divergence for logging purposes
        divergence_after_correction_field = compute_pressure_divergence(updated_velocity_field, self.mesh_info)

        return updated_velocity_field, updated_pressure_field, divergence_after_correction_field


