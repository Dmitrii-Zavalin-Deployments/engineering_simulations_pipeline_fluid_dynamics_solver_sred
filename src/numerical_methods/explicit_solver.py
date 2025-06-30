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

        # Tentative velocity field (u*)
        u_star = np.copy(velocity_field)

        # --- Advection Step ---
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        u_star[..., 0] = velocity_field[..., 0] - self.dt * advection_u
        u_star[..., 1] = velocity_field[..., 1] - self.dt * advection_v
        u_star[..., 2] = velocity_field[..., 2] - self.dt * advection_w

        # --- Diffusion Step ---
        diffusion_u = compute_diffusion_term(u_star[..., 0], self.viscosity, self.mesh_info)
        diffusion_v = compute_diffusion_term(u_star[..., 1], self.viscosity, self.mesh_info)
        diffusion_w = compute_diffusion_term(u_star[..., 2], self.viscosity, self.mesh_info)

        u_star[..., 0] += self.dt * (diffusion_u / self.density)
        u_star[..., 1] += self.dt * (diffusion_v / self.density)
        u_star[..., 2] += self.dt * (diffusion_w / self.density)

        # --- Optional External Forces ---
        # e.g., gravity or body force terms
        # u_star[..., 2] -= 9.81 * self.dt

        # --- Apply tentative BCs ---
        u_star, _ = apply_boundary_conditions(
            velocity_field=u_star,
            pressure_field=pressure_field,
            fluid_properties=self.fluid_properties_dict,
            mesh_info=self.mesh_info,
            is_tentative_step=True
        )

        # --- Pressure Projection ---
        divergence = compute_pressure_divergence(u_star, self.mesh_info)

        pressure_correction = solve_poisson_for_phi(
            divergence=divergence,
            mesh_info=self.mesh_info,
            dt=self.dt,
            tolerance=1e-6,
            max_iterations=1000
        )

        updated_velocity_field, updated_pressure_field = apply_pressure_correction(
            velocity_field=u_star,
            pressure_field=pressure_field,
            phi=pressure_correction,
            mesh_info=self.mesh_info,
            dt=self.dt,
            density=self.density
        )

        # --- Apply BCs to final velocity & pressure fields ---
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            velocity_field=updated_velocity_field,
            pressure_field=updated_pressure_field,
            fluid_properties=self.fluid_properties_dict,
            mesh_info=self.mesh_info,
            is_tentative_step=False
        )

        return updated_velocity_field, updated_pressure_field



