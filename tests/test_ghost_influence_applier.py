# tests/test_ghost_influence_applier.py
# ✅ Unit tests for ghost influence logic

import unittest
from src.grid_modules.cell import Cell
from src.physics.ghost_influence_applier import apply_ghost_influence

class TestGhostInfluenceApplier(unittest.TestCase):
    def setUp(self):
        self.spacing = (1.0, 1.0, 1.0)

    def create_cell(self, x, y, z, fluid=True, velocity=None, pressure=None):
        return Cell(
            x=x, y=y, z=z,
            velocity=velocity if velocity is not None else [0.0, 0.0, 0.0],
            pressure=pressure,
            fluid_mask=fluid
        )

    def test_basic_velocity_transfer(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[1.0, 2.0, 3.0])
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [1.0, 2.0, 3.0])

    def test_basic_pressure_transfer(self):
        fluid = self.create_cell(1.0, 1.0, 1.0, pressure=0.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, pressure=99.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.pressure, 99.0)

    def test_no_transfer_if_fluid_has_values(self):
        fluid = self.create_cell(1.0, 1.0, 1.0, velocity=[1.0, 1.0, 1.0], pressure=50.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[2.0, 2.0, 2.0], pressure=99.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 0)
        self.assertEqual(fluid.velocity, [1.0, 1.0, 1.0])
        self.assertEqual(fluid.pressure, 50.0)

    def test_multiple_ghosts_influence_same_fluid(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost1 = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[1.0, 0.0, 0.0])
        ghost2 = self.create_cell(0.0, 1.0, 1.0, fluid=False, pressure=25.0)
        grid = [fluid, ghost1, ghost2]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [1.0, 0.0, 0.0])
        self.assertEqual(fluid.pressure, 25.0)

    def test_no_influence_if_not_adjacent(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(5.0, 5.0, 5.0, fluid=False, velocity=[1.0, 1.0, 1.0], pressure=99.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 0)

    def test_tolerance_edge_case_adjacent(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.0000001, 1.0, 1.0, fluid=False, velocity=[3.0, 3.0, 3.0], pressure=12.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [3.0, 3.0, 3.0])
        self.assertEqual(fluid.pressure, 12.0)

    def test_all_fluid_cells_influenced(self):
        fluid1 = self.create_cell(1.0, 1.0, 1.0)
        fluid2 = self.create_cell(2.0, 1.0, 1.0)
        ghost = self.create_cell(1.5, 1.0, 1.0, fluid=False, velocity=[0.5, 0.5, 0.5], pressure=10.0)
        grid = [fluid1, fluid2, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 2)
        self.assertEqual(fluid1.velocity, [0.5, 0.5, 0.5])
        self.assertEqual(fluid2.pressure, 10.0)

    def test_safe_handling_of_none_velocity(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=None, pressure=20.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.pressure, 20.0)
        self.assertEqual(fluid.velocity, [0.0, 0.0, 0.0])  # unchanged

    def test_safe_handling_of_none_pressure(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[1.0, 2.0, 3.0], pressure=None)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [1.0, 2.0, 3.0])
        self.assertEqual(fluid.pressure, None)

if __name__ == "__main__":
    unittest.main()



