# src/step_2_time_stepping_loop/phase_velocity_prediction.py

from typing import Dict, Optional
import numpy as np


ORDER_6 = ("xp", "xm", "yp", "ym", "zp", "zm")


def _laplacian_vector(u_neighbors: np.ndarray,
                      u_center: np.ndarray,
                      dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Component-wise vector Laplacian using second differences.

    Parameters
    ----------
    u_neighbors : np.ndarray
        Shape (3, 6). For each component (vx, vy, vz), values at neighbors
        in order (xp, xm, yp, ym, zp, zm).
    u_center : np.ndarray
        Shape (3,). Center velocity vector [vx, vy, vz].
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    np.ndarray
        Shape (3,) with Laplacian for each component.
    """
    def second_diff(plus, minus, center, h):
        return (plus - 2.0 * center + minus) / (h * h)

    lap_vx = second_diff(u_neighbors[0, 0], u_neighbors[0, 1], u_center[0], dx) \
           + second_diff(u_neighbors[0, 2], u_neighbors[0, 3], u_center[0], dy) \
           + second_diff(u_neighbors[0, 4], u_neighbors[0, 5], u_center[0], dz)

    lap_vy = second_diff(u_neighbors[1, 0], u_neighbors[1, 1], u_center[1], dx) \
           + second_diff(u_neighbors[1, 2], u_neighbors[1, 3], u_center[1], dy) \
           + second_diff(u_neighbors[1, 4], u_neighbors[1, 5], u_center[1], dz)

    lap_vz = second_diff(u_neighbors[2, 0], u_neighbors[2, 1], u_center[2], dx) \
           + second_diff(u_neighbors[2, 2], u_neighbors[2, 3], u_center[2], dy) \
           + second_diff(u_neighbors[2, 4], u_neighbors[2, 5], u_center[2], dz)

    return np.array([lap_vx, lap_vy, lap_vz], dtype=float)


def _gradient_scalar(p_neighbors: np.ndarray,
                     dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Central-difference gradient of a scalar using 6-point stencil.

    Parameters
    ----------
    p_neighbors : np.ndarray
        Shape (6,) pressures at neighbors (xp, xm, yp, ym, zp, zm).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    np.ndarray
        grad p = [dp/dx, dp/dy, dp/dz]
    """
    dpdx = (p_neighbors[0] - p_neighbors[1]) / (2.0 * dx)
    dpdy = (p_neighbors[2] - p_neighbors[3]) / (2.0 * dy)
    dpdz = (p_neighbors[4] - p_neighbors[5]) / (2.0 * dz)
    return np.array([dpdx, dpdy, dpdz], dtype=float)


def _advection_central(u_neighbors: np.ndarray,
                       u_center: np.ndarray,
                       dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Very simple central-difference approximation for (u · ∇)u.
    This is a scaffold; real schemes (upwind/WENO) should replace it for stability.

    Parameters
    ----------
    u_neighbors : np.ndarray
        Shape (3, 6) neighbor velocities.
    u_center : np.ndarray
        Shape (3,) center velocity.
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    np.ndarray
        Approximation of convective term (u · ∇)u at center.
    """
    # Compute component-wise gradients at center
    dudx = (u_neighbors[:, 0] - u_neighbors[:, 1]) / (2.0 * dx)
    dudy = (u_neighbors[:, 2] - u_neighbors[:, 3]) / (2.0 * dy)
    dudz = (u_neighbors[:, 4] - u_neighbors[:, 5]) / (2.0 * dz)

    # (u · ∇)u = [u·∇] applied to each component: sum over directions
    u_dot_grad = u_center[0] * dudx + u_center[1] * dudy + u_center[2] * dudz
    return u_dot_grad


def _advection_upwind(u_neighbors: np.ndarray,
                      u_center: np.ndarray,
                      dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Simple first-order upwind for (u · ∇)u based on sign of center velocity.
    Still a scaffold; real implementations consider face velocities and CFL.

    Parameters
    ----------
    u_neighbors : np.ndarray
        Shape (3, 6) neighbor velocities.
    u_center : np.ndarray
        Shape (3,) center velocity.
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    np.ndarray
        Approximation of convective term (u · ∇)u at center.
    """
    # Choose directional derivatives based on velocity sign
    # X-direction
    if u_center[0] >= 0.0:
        dudx = (u_center - u_neighbors[:, 1]) / dx  # backward difference (use xm)
    else:
        dudx = (u_neighbors[:, 0] - u_center) / dx  # forward difference (use xp)

    # Y-direction
    if u_center[1] >= 0.0:
        dudy = (u_center - u_neighbors[:, 3]) / dy  # ym
    else:
        dudy = (u_neighbors[:, 2] - u_center) / dy  # yp

    # Z-direction
    if u_center[2] >= 0.0:
        dudz = (u_center - u_neighbors[:, 5]) / dz  # zm
    else:
        dudz = (u_neighbors[:, 4] - u_center) / dz  # zp

    u_dot_grad = u_center[0] * dudx + u_center[1] * dudy + u_center[2] * dudz
    return u_dot_grad


def compute_u_star(prev_state_center: Dict,
                   neighbor_states: Dict[str, Dict],
                   dt: float, rho: float, mu: float,
                   dx: float, dy: float, dz: float,
                   external_force: Optional[np.ndarray] = None,
                   advection_scheme: str = "none") -> np.ndarray:
    """
    Phase 1: Velocity prediction (u*), using diffusion, optional advection, pressure gradient, and external forces.

    Equations (component-wise scaffold, cell-centered):
        u* = u^n + (Δt/ρ) [ μ ∇² u^n - ρ · Adv(u)^n - ∇p^n + F ]

    Inputs
    ------
    prev_state_center : dict
        {
          "pressure": float,
          "velocity": {"vx": float, "vy": float, "vz": float}
        }
    neighbor_states : dict
        Keys: "xp","xm","yp","ym","zp","zm"
        Each value: {
          "pressure": float,
          "velocity": {"vx": float, "vy": float, "vz": float}
        }
    dt : float
        Time step Δt.
    rho : float
        Fluid density ρ.
    mu : float
        Dynamic viscosity μ.
    dx, dy, dz : float
        Grid spacings.
    external_force : np.ndarray, optional
        Shape (3,), body force per unit mass [Fx, Fy, Fz].
    advection_scheme : str
        "none" | "central" | "upwind". Default "none" to keep the scaffold stable.

    Returns
    -------
    np.ndarray
        u_star at center, shape (3,) corresponding to [vx*, vy*, vz*].

    Notes
    -----
    - This is a cell-centered scaffold. For a MAC (staggered) grid, adapt to face-centered velocities per component.
    - Pressure gradient uses central differences of neighbor pressures (p^n).
    - Advection term is included optionally and should be replaced with a stable scheme for production.
    """
    # Center velocity vector
    u_c = np.array([
        prev_state_center["velocity"]["vx"],
        prev_state_center["velocity"]["vy"],
        prev_state_center["velocity"]["vz"]
    ], dtype=float)

    # Neighbor velocity array (3,6)
    u_nei = np.zeros((3, 6), dtype=float)
    for idx, key in enumerate(ORDER_6):
        v = neighbor_states[key]["velocity"]
        u_nei[:, idx] = [v["vx"], v["vy"], v["vz"]]

    # Neighbor pressures (6,)
    p_nei = np.array([neighbor_states[key]["pressure"] for key in ORDER_6], dtype=float)

    # Diffusion term: μ ∇² u
    lap_u = _laplacian_vector(u_nei, u_c, dx, dy, dz)

    # Pressure gradient term: ∇p (from neighbors at time n)
    grad_p = _gradient_scalar(p_nei, dx, dy, dz)

    # Advection term: (u · ∇)u
    if advection_scheme == "central":
        adv = _advection_central(u_nei, u_c, dx, dy, dz)
    elif advection_scheme == "upwind":
        adv = _advection_upwind(u_nei, u_c, dx, dy, dz)
    elif advection_scheme == "none":
        adv = np.zeros(3, dtype=float)
    else:
        raise ValueError(f"Unsupported advection_scheme: {advection_scheme}")

    # External body force (per unit mass)
    f = external_force if external_force is not None else np.zeros(3, dtype=float)

    # Assemble predictor:
    # u* = u + (Δt/ρ)[ μ ∇² u - ρ Adv(u) - ∇p + F ]
    u_star = u_c + (dt / rho) * (mu * lap_u - rho * adv - grad_p + f)

    return u_star



