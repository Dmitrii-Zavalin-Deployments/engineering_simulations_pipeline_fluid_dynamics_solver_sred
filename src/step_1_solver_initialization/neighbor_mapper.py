# tests/test_neighbor_mapper.py
# ✅ Unit tests for src/step_1_solver_initialization/neighbor_mapper.py


from src.step_1_solver_initialization import neighbor_mapper
from src.step_1_solver_initialization.indexing_utils import grid_to_flat

# Existing functional tests remain unchanged...

def test_interior_neighbors():
    shape = (4, 3, 2)
    flat_index = grid_to_flat(2, 1, 1, shape)
    neighbors = neighbor_mapper.get_stencil_neighbors(flat_index, shape)
    assert neighbors["flat_index_i_minus_1"] == grid_to_flat(1, 1, 1, shape)
    assert neighbors["flat_index_i_plus_1"] == grid_to_flat(3, 1, 1, shape)
    assert neighbors["flat_index_j_minus_1"] == grid_to_flat(2, 0, 1, shape)
    assert neighbors["flat_index_j_plus_1"] == grid_to_flat(2, 2, 1, shape)
    assert neighbors["flat_index_k_minus_1"] == grid_to_flat(2, 1, 0, shape)
    assert neighbors["flat_index_k_plus_1"] is None

# 🧪 Debug-mode coverage test
def test_debug_output_includes_expected_lines(capsys):
    shape = (4, 3, 2)
    flat_index = grid_to_flat(0, 0, 0, shape)

    # Enable debug
    neighbor_mapper.debug = True
    neighbors = neighbor_mapper.get_stencil_neighbors(flat_index, shape)
    neighbor_mapper.debug = False  # reset

    # Capture stdout
    captured = capsys.readouterr()
    output = captured.out

    # Validate debug prints appeared
    assert "get_stencil_neighbors → flat_index=" in output
    assert "Coordinates:" in output
    assert "flat_index_i_minus_1: out of bounds" in output
    assert "flat_index_i_plus_1: valid" in output
    assert "flat_index_j_minus_1: out of bounds" in output
    assert "flat_index_j_plus_1: valid" in output
    assert "flat_index_k_minus_1: out of bounds" in output
    assert "flat_index_k_plus_1: valid" in output
    assert "Neighbor mapping complete" in output

    # Ensure functional correctness still holds
    assert neighbors["flat_index_i_minus_1"] is None
    assert neighbors["flat_index_i_plus_1"] == grid_to_flat(1, 0, 0, shape)



