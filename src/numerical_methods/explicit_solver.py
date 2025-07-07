# src/numerical_methods/explicit_solver.py

import numpy as np
import sys

try:
    from .advection import compute_advection_term
    from .diffusion import compute_diffusion_term
    from .pressure_divergence import compute_pressure_divergence
    from .multigrid_poisson import solve_poisson_multigrid
    from .pressure_correction import apply_pressure_correction
    from physics.boundary_conditions_applicator import apply_boundary_conditions
except ImportError as e:
    print(f"Error importing components for explicit_solver: {e}", file=sys.stderr)
    print("Please ensure all necessary files exist in their respective directories "
          "and contain the expected functions.", file=sys.stderr)
    sys.exit(1)


class ExplicitSolver:
    """
    Explicit fractional-step solver for incompressible Navier–Stokes equations.
    Incorporates advection, diffusion, projection, and correction stages.
    """

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        self.density = fluid_properties["density"]
        self.viscosity = fluid_properties["viscosity"]
        self.dt = dt
        self.mesh_info = mesh_info
        self.fluid_properties_dict = fluid_properties
        self.pressure_projection_passes = fluid_properties.get("pressure_projection_passes", 1)

        # Runtime health metrics
        self.last_pressure_residual = None
        self.total_divergence_before = 0.0
        self.total_divergence_after = 0.0
        self.effectiveness_score = 0.0

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray,
        smoother_iterations: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Apply BCs before velocity update
        velocity_field, pressure_field = apply_boundary_conditions(
            velocity_field,
            pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        # 2. Advection and diffusion
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        diffusion_u = compute_diffusion_term(velocity_field[..., 0], self.viscosity, self.mesh_info)
        diffusion_v = compute_diffusion_term(velocity_field[..., 1], self.viscosity, self.mesh_info)
        diffusion_w = compute_diffusion_term(velocity_field[..., 2], self.viscosity, self.mesh_info)

        # 3. Predict tentative velocity
        u_star = velocity_field[..., 0] + self.dt * (-advection_u + diffusion_u)
        v_star = velocity_field[..., 1] + self.dt * (-advection_v + diffusion_v)
        w_star = velocity_field[..., 2] + self.dt * (-advection_w + diffusion_w)
        tentative_velocity_field = np.stack((u_star, v_star, w_star), axis=-1)

        # 4. Apply BCs to tentative velocity
        tentative_velocity_field, _ = apply_boundary_conditions(
            tentative_velocity_field,
            pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=True
        )

        # 5. Projection Passes + Logging
        for pass_num in range(self.pressure_projection_passes):
            print(f"🔁 Pressure Projection Iteration {pass_num + 1}")
            divergence_before = compute_pressure_divergence(tentative_velocity_field, self.mesh_info)
            self.total_divergence_before = float(np.max(np.abs(divergence_before)))

            phi = solve_poisson_multigrid(
                divergence_before,
                self.mesh_info,
                self.dt,
                levels=3,
                smoother_iterations=smoother_iterations,
                verbose=True
            )

            # Clamp φ if needed
            if np.isnan(phi).any() or np.isinf(phi).any():
                print("⚠️ NaN/Inf in φ — clamping applied.")
                safe_max = np.finfo(np.float64).max / 10
                phi = np.clip(np.nan_to_num(phi, nan=0.0, posinf=safe_max, neginf=-safe_max), -safe_max, safe_max)

            # Apply correction
            tentative_velocity_field, pressure_field, max_div_residual = apply_pressure_correction(
                tentative_velocity_field,
                pressure_field,
                phi,
                self.mesh_info,
                self.dt,
                self.density,
                return_residual=True
            )

            # Measure effectiveness
            divergence_after = compute_pressure_divergence(tentative_velocity_field, self.mesh_info)
            self.total_divergence_after = float(np.max(np.abs(divergence_after)))
            self.effectiveness_score = 100.0 * (
                1.0 - self.total_divergence_after / max(self.total_divergence_before, 1e-8)
            )

            print(f"📉 ∇·u effectiveness: {self.effectiveness_score:.2f}% reduction")
            print(f"📏 ∇·u residual after correction: max = {max_div_residual:.4e}")
            self.last_pressure_residual = max_div_residual

        # 6. Final BC enforcement
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            tentative_velocity_field,
            pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        # 7. Final divergence field for monitoring
        divergence_field_final = compute_pressure_divergence(updated_velocity_field, self.mesh_info)

        return updated_velocity_field, updated_pressure_field, divergence_field_final



