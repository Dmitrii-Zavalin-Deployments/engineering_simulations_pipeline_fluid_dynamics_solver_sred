# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

# Import numerical method components
try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .poisson_solver import solve_poisson_for_phi
    from .pressure_correction import apply_pressure_correction
    from physics.boundary_conditions_applicator import apply_boundary_conditions
    from utils.log_utils import log_flow_metrics
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
    sys.exit(1)

class ExplicitSolver:
    """Performs a single explicit time step for incompressible fluid simulation using a fractional step method."""

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        self.density = fluid_properties["density"]
        self.viscosity = fluid_properties["viscosity"]
        self.dt = dt
        self.mesh_info = mesh_info
        self.fluid_properties_dict = fluid_properties
        self.step_counter = 0

    def step(self, velocity_field: np.ndarray, pressure_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        print("--- Starting Explicit Time Step ---")

        u_star = np.copy(velocity_field)

        print("  1. Computing advection and diffusion terms...")
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        u_star[..., 0] -= self.dt * advection_u
        u_star[..., 1] -= self.dt * advection_v
        u_star[..., 2] -= self.dt * advection_w

        diffusion_u = compute_diffusion_term(u_star[..., 0], self.viscosity, self.mesh_info)
        diffusion_v = compute_diffusion_term(u_star[..., 1], self.viscosity, self.mesh_info)
        diffusion_w = compute_diffusion_term(u_star[..., 2], self.viscosity, self.mesh_info)

        u_star[..., 0] += self.dt * (diffusion_u / self.density)
        u_star[..., 1] += self.dt * (diffusion_v / self.density)
        u_star[..., 2] += self.dt * (diffusion_w / self.density)

        if np.isnan(u_star).any() or np.isinf(u_star).any():
            print("❌ Warning: Invalid values in tentative velocity u_star — clamping to zero.")
            u_star = np.nan_to_num(u_star, nan=0.0, posinf=0.0, neginf=0.0)

        print("  2. Applying boundary conditions to the tentative velocity field (u*)...")
        u_star, _ = apply_boundary_conditions(
            velocity_field=u_star,
            pressure_field=pressure_field,
            fluid_properties=self.fluid_properties_dict,
            mesh_info=self.mesh_info,
            is_tentative_step=True
        )

        print("  3. Solving Poisson equation for pressure correction...")
        divergence = compute_pressure_divergence(u_star, self.mesh_info)

        if np.isnan(divergence).any() or np.isinf(divergence).any():
            print("❌ Warning: Invalid values in divergence field — clamping to zero.")
            divergence = np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0)

        max_div = np.max(np.abs(divergence)) if divergence.size > 0 else 0.0
        print(f"    - Max divergence before correction: {max_div:.6e}")

        phi, residual = solve_poisson_for_phi(
            divergence,
            self.mesh_info,
            self.dt,
            tolerance=1e-6,
            max_iterations=1000,
            return_residual=True
        )
        print(f"    - Pressure solver residual: {residual:.6e}")

        print("  4. Applying pressure correction to velocity and updating pressure...")
        updated_velocity, updated_pressure = apply_pressure_correction(
            u_star,
            pressure_field,
            phi,
            self.mesh_info,
            self.dt,
            self.density
        )

        if np.isnan(updated_velocity).any() or np.isinf(updated_velocity).any():
            print("❌ Warning: Invalid values in corrected velocity — clamping to zero.")
            updated_velocity = np.nan_to_num(updated_velocity, nan=0.0, posinf=0.0, neginf=0.0)

        if np.isnan(updated_pressure).any() or np.isinf(updated_pressure).any():
            print("❌ Warning: Invalid values in corrected pressure — clamping to zero.")
            updated_pressure = np.nan_to_num(updated_pressure, nan=0.0, posinf=0.0, neginf=0.0)

        print("  5. Applying final boundary conditions to the corrected fields...")
        updated_velocity, updated_pressure = apply_boundary_conditions(
            velocity_field=updated_velocity,
            pressure_field=updated_pressure,
            fluid_properties=self.fluid_properties_dict,
            mesh_info=self.mesh_info,
            is_tentative_step=False
        )

        self.step_counter += 1
        log_flow_metrics(
            velocity_field=updated_velocity,
            pressure_field=updated_pressure,
            divergence_field=divergence,
            fluid_density=self.density,
            step_count=self.step_counter,
            current_time=self.step_counter * self.dt
        )

        print("--- Explicit Time Step Complete ---")
        return updated_velocity, updated_pressure



