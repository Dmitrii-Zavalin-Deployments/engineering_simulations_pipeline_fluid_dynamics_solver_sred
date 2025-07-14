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

    def test_basic_velocity_transfer_and_tag(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[1.0, 2.0, 3.0])
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [1.0, 2.0, 3.0])
        self.assertTrue(fluid.influenced_by_ghost)

    def test_basic_pressure_transfer_and_tag(self):
        fluid = self.create_cell(1.0, 1.0, 1.0, pressure=0.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, pressure=99.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.pressure, 99.0)
        self.assertTrue(fluid.influenced_by_ghost)

    def test_no_transfer_if_fields_match(self):
        fluid = self.create_cell(1.0, 1.0, 1.0, velocity=[2.0, 2.0, 2.0], pressure=50.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[2.0, 2.0, 2.0], pressure=50.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 0)
        self.assertFalse(getattr(fluid, "influenced_by_ghost", False))

    def test_multiple_ghosts_influence_single_fluid(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost1 = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[0.0, 1.0, 0.0])
        ghost2 = self.create_cell(0.0, 1.0, 1.0, fluid=False, pressure=25.0)
        grid = [fluid, ghost1, ghost2]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [0.0, 1.0, 0.0])
        self.assertEqual(fluid.pressure, 25.0)
        self.assertTrue(fluid.influenced_by_ghost)

    def test_no_influence_if_far_apart(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(10.0, 10.0, 10.0, fluid=False, velocity=[5.0, 5.0, 5.0], pressure=80.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 0)
        self.assertFalse(getattr(fluid, "influenced_by_ghost", False))

    def test_tolerance_based_adjacency(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.00000001, 1.0, 1.0, fluid=False, velocity=[3.0, 0.0, 0.0], pressure=77.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [3.0, 0.0, 0.0])
        self.assertEqual(fluid.pressure, 77.0)
        self.assertTrue(fluid.influenced_by_ghost)

    def test_influence_multiple_fluid_cells(self):
        fluid1 = self.create_cell(1.0, 1.0, 1.0)
        fluid2 = self.create_cell(2.0, 1.0, 1.0)
        ghost = self.create_cell(1.5, 1.0, 1.0, fluid=False, velocity=[0.5, 0.5, 0.5], pressure=100.0)
        grid = [fluid1, fluid2, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 2)
        self.assertEqual(fluid1.velocity, [0.5, 0.5, 0.5])
        self.assertEqual(fluid2.pressure, 100.0)
        self.assertTrue(fluid1.influenced_by_ghost)
        self.assertTrue(fluid2.influenced_by_ghost)

    def test_none_velocity_handled_safely(self):
        fluid = self.create_cell(1.0, 1.0, 1.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=None, pressure=20.0)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [0.0, 0.0, 0.0])  # unchanged
        self.assertEqual(fluid.pressure, 20.0)
        self.assertTrue(fluid.influenced_by_ghost)

    def test_none_pressure_handled_safely(self):
        fluid = self.create_cell(1.0, 1.0, 1.0, pressure=0.0)
        ghost = self.create_cell(2.0, 1.0, 1.0, fluid=False, velocity=[1.0, 1.0, 1.0], pressure=None)
        grid = [fluid, ghost]
        count = apply_ghost_influence(grid, self.spacing)
        self.assertEqual(count, 1)
        self.assertEqual(fluid.velocity, [1.0, 1.0, 1.0])
        self.assertEqual(fluid.pressure, 0.0)  # unchanged
        self.assertTrue(fluid.influenced_by_ghost)

if __name__ == "__main__":
    unittest.main()



