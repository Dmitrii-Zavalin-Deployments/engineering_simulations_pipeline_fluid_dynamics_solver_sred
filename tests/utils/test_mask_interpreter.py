# ✅ Unit Test Suite — Mask Interpreter
# 📄 Full Path: tests/utils/test_mask_interpreter.py

import pytest
from src.utils.mask_interpreter import decode_geometry_mask_flat

def test_decode_x_major_valid():
    shape = [2, 2, 1]
    mask = [1, 0, 0, 1]  # flat order: [i + j*nx + k*nx*ny]
    result = decode_geometry_mask_flat(mask, shape, order="x-major")
    assert result == [True, False, False, True]

def test_decode_y_major_valid():
    shape = [2, 2, 1]
    mask = [1, 0, 0, 1]  # flat order: [j + i*ny + k*nx*ny]
    result = decode_geometry_mask_flat(mask, shape, order="y-major")
    assert result == [True, False, False, True]

def test_decode_z_major_valid():
    shape = [2, 2, 1]
    mask = [1, 0, 0, 1]  # flat order: [k + i*nz + j*nx*nz]
    result = decode_geometry_mask_flat(mask, shape, order="z-major")
    assert result == [True, False, False, True]

def test_custom_encoding():
    shape = [1, 1, 2]
    mask = [9, 8]
    enc = {"fluid": 9, "solid": 8}
    result = decode_geometry_mask_flat(mask, shape, encoding=enc, order="x-major")
    assert result == [True, False]

def test_invalid_shape_length_raises():
    shape = [2, 2, 1]
    mask = [1, 0, 1]  # Should be 4
    with pytest.raises(ValueError) as e:
        decode_geometry_mask_flat(mask, shape)
    assert "❌ Mask length" in str(e.value)

def test_invalid_order_raises():
    shape = [1, 1, 2]
    mask = [1, 0]
    with pytest.raises(ValueError) as e:
        decode_geometry_mask_flat(mask, shape, order="diagonal-major")
    assert "❌ Unsupported flattening order" in str(e.value)



