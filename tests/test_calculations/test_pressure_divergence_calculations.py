import numpy as np
import pytest
from src.numerical_methods.pressure_divergence import (
    compute_pressure_divergence,
    compute_pressure_gradient,
)

@pytest.fixture
def uniform_grid():
    return {
        "dx": 1.0,
        "dy": 1.0,
        "dz": 1.0,
        "grid_shape": (7, 7, 7)  # includes ghost cells: 5 interior + 2 boundaries
    }

def test_divergence_of_uniform_velocity_is_zero(uniform_grid):
    shape = uniform_grid["grid_shape"]
    velocity = np.ones(shape + (3,))  # constant velocity field
    div = compute_pressure_divergence(velocity, uniform_grid)
    assert div.shape == (5, 5, 5)
    assert np.allclose(div, 0.0)

def test_divergence_linear_increasing_x(uniform_grid):
    shape = uniform_grid["grid_shape"]
    u_field = np.zeros(shape + (3,))
    x = np.arange(shape[0])  # dx = 1.0
    u_field[..., 0] = x[:, None, None]  # ∂u/∂x = 1.0 → ∇·u = 1.0

    div = compute_pressure_divergence(u_field, uniform_grid)
    assert div.shape == (5, 5, 5)
    assert np.allclose(div, 1.0, atol=1e-6)

def test_pressure_gradient_uniform_gradient_x(uniform_grid):
    shape = uniform_grid["grid_shape"]
    x = np.arange(shape[0])  # dx = 1.0
    p = np.tile(x[:, None, None], (1, shape[1], shape[2]))

    grad = compute_pressure_gradient(p, uniform_grid)
    assert grad.shape == (5, 5, 5, 3)
    assert np.allclose(grad[..., 0], 1.0, atol=1e-6)  # ∂p/∂x = 1.0
    assert np.allclose(grad[..., 1:], 0.0, atol=1e-12)

def test_pressure_gradient_quadratic_x(uniform_grid):
    shape = uniform_grid["grid_shape"]
    x = np.linspace(-1, 1, shape[0])
    p = np.tile(x[:, None, None]**2, (1, shape[1], shape[2]))  # p(x) = x² → ∂p/∂x = 2x

    grad = compute_pressure_gradient(p, uniform_grid)
    center = grad.shape[0] // 2
    g_center = grad[center, center, center, 0]
    assert g_center == pytest.approx(0.0, abs=1e-1)

    left = grad[0, center, center, 0]
    right = grad[-1, center, center, 0]
    assert left < 0  # gradient negative on left side of parabola
    assert right > 0  # gradient positive on right side

def test_pressure_gradient_3d_paraboloid(uniform_grid):
    shape = uniform_grid["grid_shape"]
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    z = np.linspace(-1, 1, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    p = X**2 + Y**2 + Z**2

    grad = compute_pressure_gradient(p, uniform_grid)
    center = grad.shape[0] // 2
    gx, gy, gz = grad[center, center, center]
    assert gx == pytest.approx(0.0, abs=1e-1)
    assert gy == pytest.approx(0.0, abs=1e-1)
    assert gz == pytest.approx(0.0, abs=1e-1)

    edge = grad[0, -1, -1]
    assert edge[0] < 0  # ∂p/∂x negative at min x
    assert edge[1] > 0  # ∂p/∂y positive at max y
    assert edge[2] > 0  # ∂p/∂z positive at max z



