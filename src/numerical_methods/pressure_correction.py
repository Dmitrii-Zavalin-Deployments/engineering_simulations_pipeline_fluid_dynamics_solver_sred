# src/numerical_methods/pressure_correction.py
import numpy as np

def calculate_gradient(field, h, axis):
    """
    Calculates the gradient of a field using central differencing.
    Assumes the field has the shape of the physical grid (nx, ny, nz)
    and pads it to enable central differencing at boundaries.

    Args:
        field (np.ndarray): The field to calculate the gradient of (nx, ny, nz).
        h (float): Grid spacing (dx, dy, or dz).
        axis (int): The axis along which to calculate the gradient (0=x, 1=y, 2=z).

    Returns:
        np.ndarray: The gradient of the field along the specified axis (nx, ny, nz).
    """
    # Pad the field with a layer of ghost cells using 'edge' values to allow central differencing
    padded_field = np.pad(field, ((1, 1), (1, 1), (1, 1)), mode='edge')
    
    # Central differencing on the padded field
    if axis == 0:
        return (padded_field[2:, 1:-1, 1:-1] - padded_field[:-2, 1:-1, 1:-1]) / (2 * h)
    elif axis == 1:
        return (padded_field[1:-1, 2:, 1:-1] - padded_field[1:-1, :-2, 1:-1]) / (2 * h)
    elif axis == 2:
        return (padded_field[1:-1, 1:-1, 2:] - padded_field[1:-1, 1:-1, :-2]) / (2 * h)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

def apply_pressure_correction(velocity_next, p_field, phi, mesh_info, time_step, density):
    """
    Corrects the tentative velocity field to be divergence-free
    using the calculated pressure potential (phi).

    Args:
        velocity_next (np.ndarray): The tentative velocity field with ghost cells (nx+2, ny+2, nz+2, 3).
        p_field (np.ndarray): The pressure field with ghost cells (nx+2, ny+2, nz+2).
        phi (np.ndarray): The pressure potential calculated from the Poisson solver (nx, ny, nz).
        mesh_info (dict): Grid information including dx, dy, dz.
        time_step (float): The time step (dt).
        density (float): The fluid density.

    Returns:
        tuple: A tuple containing the corrected velocity field and the updated pressure field.
    """
    dx, dy, dz = mesh_info['dx'], mesh_info['dy'], mesh_info['dz']

    # Update the pressure field by adding the potential phi to the interior cells
    # This maps the (nx, ny, nz) phi field to the (nx+2, ny+2, nz+2) pressure field.
    updated_pressure_field = p_field.copy()
    updated_pressure_field[1:-1, 1:-1, 1:-1] += phi

    # Calculate gradients of the pressure potential phi
    # These gradients will have the shape of the physical grid (nx, ny, nz)
    dphi_dx = calculate_gradient(phi, dx, axis=0)
    dphi_dy = calculate_gradient(phi, dy, axis=1)
    dphi_dz = calculate_gradient(phi, dz, axis=2)

    # Create a copy of the velocity field to apply corrections to
    corrected_velocity_field = velocity_next.copy()
    
    # --- FIX: Apply pressure correction only to the interior grid cells ---
    # The pressure gradients (dphi_d*) have the shape of the physical grid (nx, ny, nz).
    # The velocity field has ghost cells (nx+2, ny+2, nz+2, 3).
    # We must slice the velocity field to match the shape of the gradients before the update.
    # The interior slice for the ghosted grid is [1:-1, 1:-1, 1:-1].
    
    # X-component (Vx)
    corrected_velocity_field[1:-1, 1:-1, 1:-1, 0] -= time_step * dphi_dx / density
    
    # Y-component (Vy)
    corrected_velocity_field[1:-1, 1:-1, 1:-1, 1] -= time_step * dphi_dy / density
    
    # Z-component (Vz)
    corrected_velocity_field[1:-1, 1:-1, 1:-1, 2] -= time_step * dphi_dz / density
    # --- END FIX ---
    
    return corrected_velocity_field, updated_pressure_field
