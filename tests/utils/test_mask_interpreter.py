# tests/utils/test_mask_interpreter.py
# 🧪 Validates geometry mask flattening across orders, shapes, encodings

import pytest
from src.utils.mask_interpreter import decode_geometry_mask_flat, decode_fluid_mask

def test_x_major_flattening_correctness():
    flat_mask = [1, 0, 1, 1]
    shape = [2, 2, 1]
    result = decode_geometry_mask_flat(flat_mask, shape, order="x-major")
    assert result == [True, False, True, True]

def test_y_major_flattening_correctness():
    flat_mask = [0, 1, 1, 1]
    shape = [2, 2, 1]
    result = decode_geometry_mask_flat(flat_mask, shape, order="y-major")
    assert result == [False, True, True, True]

def test_z_major_flattening_correctness():
    flat_mask = [1, 0, 0, 1]
    shape = [2, 2, 1]
    result = decode_geometry_mask_flat(flat_mask, shape, order="z-major")
    assert result == [True, False, False, True]

def test_3d_x_major_flattening():
    flat_mask = [1, 0, 1, 0, 0, 1, 1, 1]
    shape = [2, 2, 2]
    result = decode_geometry_mask_flat(flat_mask, shape, order="x-major")
    assert len(result) == 8
    assert result == [True, False, True, False, False, True, True, True]

def test_custom_encoding_values():
    flat_mask = [9, 2, 9, 2]
    shape = [2, 2, 1]
    encoding = {"fluid": 9, "solid": 2}
    result = decode_geometry_mask_flat(flat_mask, shape, encoding=encoding)
    assert result == [True, False, True, False]

def test_shape_length_mismatch_raises():
    flat_mask = [1, 0, 1]
    shape = [2, 2, 1]
    with pytest.raises(ValueError, match="Mask length 3 does not match expected shape"):
        decode_geometry_mask_flat(flat_mask, shape)

def test_unsupported_order_raises():
    flat_mask = [1, 0, 1, 1]
    shape = [2, 2, 1]
    with pytest.raises(ValueError, match="Unsupported flattening order"):
        decode_geometry_mask_flat(flat_mask, shape, order="unknown-order")

def test_empty_mask_and_shape_zero():
    with pytest.raises(ValueError):
        decode_geometry_mask_flat([], [0, 0, 0])

def test_empty_mask_in_fluid_decoder_raises():
    bad_mask = {
        "geometry_mask_shape": [0, 0, 0],
        "geometry_mask_flat": [],
        "mask_encoding": {"fluid": 1, "solid": 0},
        "flattening_order": "x-major"
    }
    with pytest.raises(ValueError, match="Mask length 0 does not match expected shape"):
        decode_fluid_mask(bad_mask, domain_resolution=(2, 1, 1))

def test_all_solid_returns_false_only():
    flat_mask = [0] * 6
    shape = [3, 2, 1]
    result = decode_geometry_mask_flat(flat_mask, shape)
    assert result == [False] * 6



