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
    An explicit solver for the incompressible Navier–Stokes equations
    using a fractional step (Projection) method.
    """

    def __init__(self, fluid_properties: dict, mesh_info: dict, dt: float):
        self.density = fluid_properties['density']
        self.viscosity = fluid_properties['viscosity']
        self.dt = dt
        self.mesh_info = mesh_info
        self.fluid_properties_dict = fluid_properties
        self.pressure_projection_passes = fluid_properties.get("pressure_projection_passes", 1)
        self.last_pressure_residual = None

    def step(
        self,
        velocity_field: np.ndarray,
        pressure_field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Apply initial BCs
        velocity_field, pressure_field = apply_boundary_conditions(
            velocity_field,
            pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        # 2. Compute advection + diffusion
        advection_u = compute_advection_term(velocity_field[..., 0], velocity_field, self.mesh_info)
        advection_v = compute_advection_term(velocity_field[..., 1], velocity_field, self.mesh_info)
        advection_w = compute_advection_term(velocity_field[..., 2], velocity_field, self.mesh_info)

        diffusion_u = compute_diffusion_term(velocity_field[..., 0], self.viscosity, self.mesh_info)
        diffusion_v = compute_diffusion_term(velocity_field[..., 1], self.viscosity, self.mesh_info)
        diffusion_w = compute_diffusion_term(velocity_field[..., 2], self.viscosity, self.mesh_info)

        # 3. Predict tentative velocity (u_star)
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

        # 5–6. Pressure projection loop
        for pass_num in range(self.pressure_projection_passes):
            print(f"🔁 Pressure Projection Iteration {pass_num + 1}")
            divergence = compute_pressure_divergence(tentative_velocity_field, self.mesh_info)

            phi = solve_poisson_multigrid(
                divergence,
                self.mesh_info,
                self.dt,
                levels=3
            )

            if np.any(np.isnan(phi)) or np.any(np.isinf(phi)):
                print("⚠️ NaN or Inf in pressure correction φ — clamping.")
                max_val = np.finfo(np.float64).max / 10
                phi = np.clip(phi, -max_val, max_val)

            tentative_velocity_field, pressure_field, max_div_residual = apply_pressure_correction(
                tentative_velocity_field,
                pressure_field,
                phi,
                self.mesh_info,
                self.dt,
                self.density,
                return_residual=True
            )

            print(f"📏 ∇·u residual after correction: max={max_div_residual:.4e}")
            self.last_pressure_residual = max_div_residual

        # 7. Final BC enforcement
        updated_velocity_field, updated_pressure_field = apply_boundary_conditions(
            tentative_velocity_field,
            pressure_field,
            self.fluid_properties_dict,
            self.mesh_info,
            is_tentative_step=False
        )

        # 8. Divergence audit
        divergence_after_correction_field = compute_pressure_divergence(
            updated_velocity_field,
            self.mesh_info
        )

        return updated_velocity_field, updated_pressure_field, divergence_after_correction_field



