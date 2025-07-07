# src/utils/log_utils.py

import numpy as np
import logging

# 📝 Configure structured simulation logger
logger = logging.getLogger("simulation_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_flow_metrics(
    velocity_field: np.ndarray,
    pressure_field: np.ndarray,
    divergence_field: np.ndarray,
    fluid_density: float,
    step_count: int,
    current_time: float,
    output_frequency_steps: int,
    num_steps: int,
    dt: float = None,
    cfl: float = None,
    residual_divergence: float = None,
    event_tag: str = None,
    recovery_triggered: bool = False
):
    """
    Logs diagnostic metrics for simulation stability and runtime behavior.

    Args:
        velocity_field (np.ndarray): Velocity [nx+2, ny+2, nz+2, 3]
        pressure_field (np.ndarray): Pressure [nx+2, ny+2, nz+2]
        divergence_field (np.ndarray): Divergence ∇·u field
        fluid_density (float): Fluid density
        step_count (int): Current step number
        current_time (float): Simulation time in seconds
        output_frequency_steps (int): Step interval for diagnostics
        num_steps (int): Total number of simulation steps
        dt (float): Time step
        cfl (float): CFL number
        residual_divergence (float): Post-correction divergence residual
        event_tag (str): Optional event label
        recovery_triggered (bool): Whether recovery logic was applied
    """
    interior_v = velocity_field[1:-1, 1:-1, 1:-1, :]
    interior_p = pressure_field[1:-1, 1:-1, 1:-1]
    interior_div = divergence_field[1:-1, 1:-1, 1:-1]

    if interior_v.size == 0 or interior_p.size == 0:
        logger.warning(f"Empty interior domain @ step {step_count} — diagnostics skipped.")
        return

    # 🧼 Sanitize for safety
    interior_v = np.nan_to_num(interior_v, nan=0.0, posinf=0.0, neginf=0.0)
    interior_p = np.nan_to_num(interior_p, nan=0.0, posinf=0.0, neginf=0.0)
    interior_div = np.nan_to_num(interior_div, nan=0.0, posinf=0.0, neginf=0.0)

    if (step_count % output_frequency_steps == 0) or \
       (step_count == num_steps and step_count != 0):

        velocity_mag = np.linalg.norm(interior_v, axis=-1)
        kinetic_energy = 0.5 * fluid_density * np.sum(velocity_mag ** 2)
        max_velocity = np.max(velocity_mag)

        min_p = np.min(interior_p)
        max_p = np.max(interior_p)
        mean_p = np.mean(interior_p)
        std_p = np.std(interior_p)

        max_div = float(np.max(np.abs(interior_div))) if interior_div.size > 0 else 0.0
        mean_div = float(np.mean(np.abs(interior_div))) if interior_div.size > 0 else 0.0

        logger.info(f"📊 Step {step_count} @ t = {current_time:.4f}s")
        logger.info(f"    • Kinetic Energy              : {kinetic_energy:.4e}")
        logger.info(f"    • Max Velocity Magnitude      : {max_velocity:.4e}")
        logger.info(f"    • Pressure Range              : [{min_p:.4e}, {max_p:.4e}]")
        logger.info(f"    • Mean Pressure               : {mean_p:.4e}")
        logger.info(f"    • Pressure Std Dev            : {std_p:.4e}")
        logger.info(f"    • Divergence ∇·u              : Max = {max_div:.4e}, Mean = {mean_div:.4e}")

        if dt is not None:
            logger.info(f"    • Time Step (dt)              : {dt:.4e}")
        if cfl is not None:
            logger.info(f"    • CFL Number                  : {cfl:.4e}")
        if residual_divergence is not None:
            logger.info(f"    • Post-Correction Residual ∇·u: {residual_divergence:.4e}")
        if event_tag:
            logger.info(f"🛠 Event Triggered                : {event_tag}")
        if recovery_triggered:
            logger.info(f"🔧 Adaptive Recovery Applied      : ✅ Projection clamped, dt reduced")



