# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

# Import the individual numerical methods you already have
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    from .pressure_correction import apply_pressure_correction
    # This is crucial for applying boundary conditions
    from physics.boundary_conditions_applicator import apply_boundary_conditions
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
    print("Please ensure advection.py, diffusion.py, pressure_divergence.py, "
          "poisson_solver.py, pressure_correction.py, and boundary_conditions.py exist in their respective directories "
          "and contain the expected functions.", file=sys.stderr)
    sys.exit(1)


class ExplicitSolver:
    """
    Orchestrates a single explicit time step for a fluid simulation using a fractional
    step method (advection -> diffusion -> pressure projection).

    This class encapsulates the solver's parameters (dt, viscosity, density, mesh_info)
    and provides a `step` method to advance the fluid state.
    """

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        """
        Initializes the explicit solver with simulation parameters.

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
        
        # This dictionary is created once in the constructor
        self.fluid_properties_dict = fluid_properties

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs one explicit time step on the fluid fields.

        Args:
            velocity_field (np.ndarray): Current velocity field (U, V, W components)
                                         Shape: (nx, ny, nz, 3).
            pressure_field (np.ndarray): Current pressure field. Shape: (nx, ny, nz).

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated velocity_field and pressure_field after one time step.
        """

        # Start with a copy to avoid modifying the input field prematurely
        u_star = np.copy(velocity_field)

        # 1. Advection Step
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        u_star[..., 0] = velocity_field[..., 0] - self.dt * advection_u
        u_star[..., 1] = velocity_field[..., 1] - self.dt * advection_v
        u_star[..., 2] = velocity_field[..., 2] - self.dt * advection_w

        # 2. Diffusion Step
        diffusion_u = compute_diffusion_term(u_star[..., 0], self.viscosity, self.mesh_info)
        diffusion_v = compute_diffusion_term(u_star[..., 1], self.viscosity, self.mesh_info)
        diffusion_w = compute_diffusion_term(u_star[..., 2], self.viscosity, self.mesh_info)

        u_star[..., 0] += self.dt * (diffusion_u / self.density)
        u_star[..., 1] += self.dt * (diffusion_v / self.density)
        u_star[..., 2] += self.dt * (diffusion_w / self.density)

        # 3. Add External Forces (if any)
        # u_star[..., 2] -= 9.81 * self.dt # Example for gravity in -z direction

        # --- Apply Boundary Conditions to the intermediate velocity field (u_star) ---
        # u_star is a "tentative" velocity field, so is_tentative_step should be True.
        u_star, _ = apply_boundary_conditions(
            u_star,
            pressure_field,
            self.fluid_properties_dict, # Pass the properties dictionary
            self.mesh_info,
            is_tentative_step=True # Crucial fix: indicate this is a tentative step
        )

        # --- Step 2: Pressure Projection (ensures incompressibility) ---
        # Compute the divergence of the intermediate velocity field
        divergence = compute_pressure_divergence(u_star, self.mesh_info)

        # Solve Poisson equation for pressure correction (phi)
        poisson_tolerance = 1e-6
        poisson_max_iter = 1000

        pressure_correction = solve_poisson_for_phi(
            divergence,
            self.mesh_info,
            self.dt,
            tolerance=poisson_tolerance,
            max_iterations=poisson_max_iter
        )

        # Apply pressure correction
        updated_velocity_field, updated_pressure_field = apply_pressure_correction(
            u_star,
            pressure_field,
            pressure_correction,
            self.mesh_info,
            self.dt,
            self.density
        )

        # --- Apply Boundary Conditions to the final updated fields after pressure correction ---
        # This ensures the final fields conform to boundary conditions.
        # These are the "final" fields for this time step, so is_tentative_step should be False.
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            updated_velocity_field,
            updated_pressure_field,
            self.fluid_properties_dict, # Pass the properties dictionary
            self.mesh_info,
            is_tentative_step=False # Crucial fix: indicate this is a final step
        )

        return updated_velocity_field, updated_pressure_field