import numpy as np
import pytest

def generate_random_velocity_field(shape, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform(-1.0, 1.0, shape)
    v = np.random.uniform(-1.0, 1.0, shape)
    w = np.random.uniform(-1.0, 1.0, shape)
    return u, v, w

def compute_divergence(u, v, w, dx, dy, dz):
    div = (
        (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx) +
        (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
        (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)
    )
    return div

@pytest.mark.parametrize("grid_shape", [(10, 10, 10), (8, 12, 6), (4, 4, 4)])
def test_random_velocity_field_divergence_small(grid_shape):
    padded_shape = tuple(s + 2 for s in grid_shape)
    dx, dy, dz = 1.0, 1.0, 1.0  # uniform grid spacing

    u, v, w = generate_random_velocity_field(padded_shape, seed=42)
    divergence = compute_divergence(u, v, w, dx, dy, dz)

    mean_div = np.mean(np.abs(divergence))
    max_div = np.max(np.abs(divergence))

    print(f"Grid shape {grid_shape}: Mean divergence = {mean_div:.4e}, Max divergence = {max_div:.4e}")

    # Assert that the raw divergence is non-trivially large before projection
    assert mean_div > 1e-3
    assert max_div > 1e-2



