# tests/test_mask_interpreter.py
# 🧪 Unit tests for decode_geometry_mask_flat()

import pytest
from src.utils.mask_interpreter import decode_geometry_mask_flat

def test_x_major_flattening():
    flat = [1, 0, 1, 0, 1, 0]
    shape = [3, 2, 1]  # nx, ny, nz
    expected = [True, False, True, False, True, False]
    result = decode_geometry_mask_flat(flat, shape, {"fluid": 1, "solid": 0}, "x-major")
    assert result == expected


def test_y_major_flattening():
    flat = [0, 1, 1, 0, 0, 1]
    shape = [3, 2, 1]
    expected = [False, True, True, False, False, True]
    result = decode_geometry_mask_flat(flat, shape, {"fluid": 1}, "y-major")
    assert result == expected


def test_z_major_flattening():
    flat = [1, 0, 1, 0, 1, 0]
    shape = [3, 2, 1]
    expected = [True, False, True, False, True, False]
    result = decode_geometry_mask_flat(flat, shape, {"fluid": 1}, "z-major")
    assert result == expected


def test_custom_encoding():
    flat = ["f", "s", "s", "f"]
    shape = [2, 2, 1]
    encoding = {"fluid": "f", "solid": "s"}
    expected = [True, False, False, True]
    result = decode_geometry_mask_flat(flat, shape, encoding, "x-major")
    assert result == expected


def test_invalid_shape_length():
    flat = [1, 0, 1]
    shape = [2, 2, 1]  # Should require 4 entries
    with pytest.raises(ValueError, match="Mask length"):
        decode_geometry_mask_flat(flat, shape)


def test_invalid_order():
    flat = [1, 1]
    shape = [2, 1, 1]
    with pytest.raises(ValueError, match="Unsupported flattening order"):
        decode_geometry_mask_flat(flat, shape, {"fluid": 1}, "invalid-order")


def test_default_encoding():
    flat = [1, 0, 0, 1]
    shape = [2, 2, 1]
    expected = [True, False, False, True]
    result = decode_geometry_mask_flat(flat, shape)  # Uses default encoding
    assert result == expected


def test_nonstandard_types():
    flat = [True, False, False, True]
    shape = [2, 2, 1]
    encoding = {"fluid": True, "solid": False}
    expected = [True, False, False, True]
    result = decode_geometry_mask_flat(flat, shape, encoding, "x-major")
    assert result == expected



