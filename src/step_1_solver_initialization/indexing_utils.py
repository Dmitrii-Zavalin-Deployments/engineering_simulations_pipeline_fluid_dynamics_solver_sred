# tests/test_indexing_utils.py
# ✅ Unit tests for src/step_1_solver_initialization/indexing_utils.py


from src.step_1_solver_initialization import indexing_utils

# Existing functional tests remain unchanged...

def test_grid_to_flat_and_flat_to_grid_roundtrip():
    shape = (4, 3, 2)
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                flat = indexing_utils.grid_to_flat(x, y, z, shape)
                coords = indexing_utils.flat_to_grid(flat, shape)
                assert coords == [x, y, z]

def test_is_valid_grid_index_and_flat_index():
    shape = (4, 3, 2)
    assert indexing_utils.is_valid_grid_index(0, 0, 0, shape)
    assert not indexing_utils.is_valid_grid_index(-1, 0, 0, shape)
    assert indexing_utils.is_valid_flat_index(0, shape)
    assert not indexing_utils.is_valid_flat_index(24, shape)

# 🧪 Debug-mode coverage tests
def test_debug_output_grid_to_flat(capsys):
    shape = (4, 3, 2)
    indexing_utils.debug = True
    flat = indexing_utils.grid_to_flat(3, 2, 1, shape)
    indexing_utils.debug = False
    captured = capsys.readouterr()
    output = captured.out
    assert "grid_to_flat" in output
    assert "Formula" in output
    assert "Result" in output
    assert flat == 23

def test_debug_output_flat_to_grid(capsys):
    shape = (4, 3, 2)
    indexing_utils.debug = True
    coords = indexing_utils.flat_to_grid(23, shape)
    indexing_utils.debug = False
    captured = capsys.readouterr()
    output = captured.out
    assert "flat_to_grid" in output
    assert "Reverse mapping formulas:" in output
    assert "Result" in output
    assert coords == [3, 2, 1]

def test_debug_output_is_valid_grid_index(capsys):
    shape = (4, 3, 2)
    indexing_utils.debug = True
    valid = indexing_utils.is_valid_grid_index(0, 0, 0, shape)
    indexing_utils.debug = False
    captured = capsys.readouterr()
    output = captured.out
    assert "is_valid_grid_index" in output
    assert "→ True" in output
    assert valid is True

def test_debug_output_is_valid_flat_index(capsys):
    shape = (4, 3, 2)
    indexing_utils.debug = True
    valid = indexing_utils.is_valid_flat_index(23, shape)
    indexing_utils.debug = False
    captured = capsys.readouterr()
    output = captured.out
    assert "is_valid_flat_index" in output
    assert "→ True" in output
    assert valid is True



