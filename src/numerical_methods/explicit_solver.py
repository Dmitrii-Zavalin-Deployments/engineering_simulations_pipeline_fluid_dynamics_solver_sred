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
    from physics.boundary_conditions_applicator import apply_boundary_conditions
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
    print("Please ensure advection.py, diffusion.py, pressure_divergence.py, "
          "poisson_solver.py, pressure_correction.py, and boundary_conditions_applicator.py exist and are properly implemented.",
          file=sys.stderr)
    sys.exit(1)


class ExplicitSolver:
    """
    Orchestrates a single explicit time step for a fluid simulation using a fractional
    step method (advection -> diffusion -> pressure projection).
    """

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        self.density = fluid_properties["density"]
        self.viscosity = fluid_properties["viscosity"]
        self.dt = dt
        self.mesh_info = mesh_info
        self.fluid_properties_dict = fluid_properties

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs one explicit time step on the fluid fields.

        Args:
            velocity_field: Current velocity field (shape: [nx, ny, nz, 3])
            pressure_field: Current pressure field (shape: [nx, ny, nz])

        Returns:
            A tuple of (updated_velocity_field, updated_pressure_field)
        """
        print("--- Starting Explicit Time Step ---")

        # Create a copy of the velocity field to compute the tentative velocity (u*)
        u_star = np.copy(velocity_field)

        # --- Step 1: Advection and Diffusion ---
        print("  1. Computing advection and diffusion terms...")
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        # Update velocity with advection term
        u_star[..., 0] -= self.dt * advection_u
        u_star[..., 1] -= self.dt * advection_v
        u_star[..., 2] -= self.dt * advection_w

        diffusion_u = compute_diffusion_term(u_star[..., 0], self.viscosity, self.mesh_info)
        diffusion_v = compute_diffusion_term(u_star[..., 1], self.viscosity, self.mesh_info)
        diffusion_w = compute_diffusion_term(u_star[..., 2], self.viscosity, self.mesh_info)
        
        # Update velocity with diffusion term
        u_star[..., 0] += self.dt * (diffusion_u / self.density)
        u_star[..., 1] += self.dt * (diffusion_v / self.density)
        u_star[..., 2] += self.dt * (diffusion_w / self.density)

        # --- Optional External Forces ---
        # u_star[..., 2] -= 9.81 * self.dt

        # --- Step 2: Apply BCs to the tentative velocity field (u*) ---
        # This is where no-slip velocity boundaries are enforced for the intermediate velocity.
        print("  2. Applying boundary conditions to the tentative velocity field (u*)...")
        u_star, _ = apply_boundary_conditions(
            velocity_field=u_star,
            pressure_field=pressure_field, # The pressure field is not updated at this stage
            fluid_properties=self.fluid_properties_dict,
            mesh_info=self.mesh_info,
            is_tentative_step=True # Flag to apply only velocity BCs
        )

        # --- Step 3: Pressure Projection (Solve Poisson Equation) ---
        print("  3. Solving Poisson equation for pressure correction...")
        divergence = compute_pressure_divergence(u_star, self.mesh_info)
        
        # --- Add useful logging for debugging divergence ---
        max_div = np.max(np.abs(divergence))
        print(f"    - Max divergence before correction: {max_div:.6e}")
        
        # The solve_poisson_for_phi function now takes the mesh_info to apply
        # the Dirichlet pressure BCs as constraints during the solve.
        pressure_correction, residual = solve_poisson_for_phi(
            divergence,
            self.mesh_info,
            self.dt,
            tolerance=1e-6,
            max_iterations=1000,
            return_residual=True # Request the residual for logging
        )
        
        # --- Add logging for pressure solver residual ---
        print(f"    - Pressure solver residual: {residual:.6e}")

        # --- Step 4: Correct the velocity field and update pressure ---
        print("  4. Applying pressure correction to velocity and updating pressure...")
        updated_velocity_field, updated_pressure_field = apply_pressure_correction(
            u_star,
            pressure_field,
            pressure_correction,
            self.mesh_info,
            self.dt,
            self.density
        )

        # --- Step 5: Apply final BCs to the corrected fields ---
        # This is the crucial step for enforcing Dirichlet pressure conditions.
        print("  5. Applying final boundary conditions to the corrected fields...")
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            velocity_field=updated_velocity_field,
            pressure_field=updated_pressure_field,
            fluid_properties=self.fluid_properties_dict,
            mesh_info=self.mesh_info,
            is_tentative_step=False # Flag to apply both velocity and pressure BCs
        )

        # --- Add optional logging for flow metrics ---
        # This is extremely useful for tracking simulation progress and convergence.
        interior_velocity = updated_velocity_field[1:-1, 1:-1, 1:-1, :]
        total_kinetic_energy = 0.5 * self.density * np.sum(np.linalg.norm(interior_velocity, axis=-1)**2)
        print(f"    - Total Kinetic Energy: {total_kinetic_energy:.4e}")
        max_velocity_magnitude = np.max(np.linalg.norm(interior_velocity, axis=-1))
        print(f"    - Max Velocity Magnitude: {max_velocity_magnitude:.4e}")
        
        interior_pressure = updated_pressure_field[1:-1, 1:-1, 1:-1]
        print(f"    - Pressure range (interior): [{np.min(interior_pressure):.4e}, {np.max(interior_pressure):.4e}]")

        print("--- Explicit Time Step Complete ---")
        return updated_velocity_field, updated_pressure_field


