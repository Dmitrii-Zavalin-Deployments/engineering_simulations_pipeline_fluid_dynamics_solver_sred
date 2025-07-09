# tests/utils/test_utils_velocity.py

import numpy as np

def generate_velocity_with_divergence(grid_shape, pattern="x-ramp", scale=1.0, seed=None):
    """
    Creates a 3D velocity field with a specified divergence pattern.

    Args:
        grid_shape (tuple): 3D grid shape (nx, ny, nz), including ghost cells.
        pattern (str): One of ['x-ramp', 'random', 'radial', 'checkerboard'].
        scale (float): Scaling factor for intensity.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Velocity field of shape (nx, ny, nz, 3)
    """
    u = np.zeros(grid_shape + (3,), dtype=np.float64)

    if pattern == "x-ramp":
        for i in range(1, grid_shape[0] - 1):
            u[i, :, :, 0] = scale * i
    elif pattern == "checkerboard":
        x = np.arange(grid_shape[0])[:, None, None]
        y = np.arange(grid_shape[1])[None, :, None]
        z = np.arange(grid_shape[2])[None, None, :]
        mask = ((x + y + z) % 2 == 0)
        u[mask, 0] = scale
    elif pattern == "random":
        if seed is not None:
            np.random.seed(seed)
        u = np.random.randn(*grid_shape, 3) * scale
    elif pattern == "radial":
        x = np.linspace(-1, 1, grid_shape[0])
        y = np.linspace(-1, 1, grid_shape[1])
        z = np.linspace(-1, 1, grid_shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        R = np.sqrt(X**2 + Y**2 + Z**2) + 1e-8
        u[..., 0] = scale * X / R
        u[..., 1] = scale * Y / R
        u[..., 2] = scale * Z / R
    else:
        raise ValueError(f"Unknown pattern '{pattern}'.")

    return u



