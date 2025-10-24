# tests/grid_modules/test_boundary_manager.py
# ✅ Validation suite for src/grid_modules/boundary_manager.py

from src.grid_modules.boundary_manager import apply_boundaries
from src.grid_modules.cell import Cell

class MockCell(Cell):
    def __init__(self, x, y, z):
        super().__init__(x=x, y=y, z=z, velocity=[0, 0, 0], pressure=0.0, fluid_mask=True)

def test_apply_boundaries_returns_original_cells():
    domain = {"nx": 4, "ny": 4, "nz": 4}
    cells = [MockCell(x, y, z) for x in range(4) for y in range(4) for z in range(4)]
    result = apply_boundaries(cells, domain)
    assert result == cells  # Function is non-mutating and returns original list

def test_apply_boundaries_prints_expected_faces(capsys):
    domain = {"nx": 3, "ny": 5, "nz": 2}
    cells = [MockCell(0, 0, 0)]
    apply_boundaries(cells, domain)
    output = capsys.readouterr().out
    assert "nx=3, ny=5, nz=2" in output
    assert "x={0, 2}" in output
    assert "y={0, 4}" in output
    assert "z={0, 1}" in output

def test_apply_boundaries_handles_zero_resolution():
    domain = {"nx": 0, "ny": 0, "nz": 0}
    cells = [MockCell(0, 0, 0)]
    result = apply_boundaries(cells, domain)
    assert result == cells  # Should still return original list



