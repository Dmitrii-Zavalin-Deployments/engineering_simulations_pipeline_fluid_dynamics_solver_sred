import pytest
from src.utils.mask_interpreter import decode_geometry_mask_flat

def test_x_major_decoding():
    flat_mask = [1, 0, 1, 0, 1, 0, 1, 0]  # 2x2x2 grid
    shape = [2, 2, 2]
    result = decode_geometry_mask_flat(flat_mask, shape, order="x-major")
    assert result == [True, False, True, False, True, False, True, False]

def test_y_major_decoding():
    flat_mask = [1, 0, 1, 0, 1, 0, 1, 0]  # 2x2x2 grid
    shape = [2, 2, 2]
    result = decode_geometry_mask_flat(flat_mask, shape, order="y-major")
    assert result == [True, False, True, False, True, False, True, False]

def test_z_major_decoding():
    flat_mask = [1, 0, 1, 0, 1, 0, 1, 0]  # 2x2x2 grid
    shape = [2, 2, 2]
    result = decode_geometry_mask_flat(flat_mask, shape, order="z-major")
    assert result == [True, True, False, False, True, True, False, False]

def test_custom_encoding():
    flat_mask = [9, 4, 9, 4]
    shape = [2, 2, 1]
    encoding = {"fluid": 9, "solid": 4}
    result = decode_geometry_mask_flat(flat_mask, shape, encoding=encoding, order="x-major")
    assert result == [True, False, True, False]

def test_shape_mismatch_raises():
    flat_mask = [1, 0, 1]
    shape = [2, 2, 1]  # Expected length = 4
    with pytest.raises(ValueError) as e:
        decode_geometry_mask_flat(flat_mask, shape)
    assert "does not match expected shape" in str(e.value)

def test_invalid_order_raises():
    flat_mask = [1, 0, 1, 0]
    shape = [2, 2, 1]
    with pytest.raises(ValueError) as e:
        decode_geometry_mask_flat(flat_mask, shape, order="invalid-order")
    assert "Unsupported flattening order" in str(e.value)

def test_all_fluid_cells():
    flat_mask = [1] * 8
    shape = [2, 2, 2]
    result = decode_geometry_mask_flat(flat_mask, shape)
    assert all(result)

def test_all_solid_cells():
    flat_mask = [0] * 8
    shape = [2, 2, 2]
    result = decode_geometry_mask_flat(flat_mask, shape)
    assert not any(result)



