# tests/test_divergence_tracker.py
# ✅ Unit tests for divergence tracking utilities

import unittest
import tempfile
import os
from src.grid_modules.cell import Cell
from src.utils.divergence_tracker import (
    compute_divergence,
    compute_divergence_stats,
    dump_divergence_map
)

class TestDivergenceTracker(unittest.TestCase):
    def setUp(self):
        self.spacing = (1.0, 1.0, 1.0)
        self.config = {"reflex_verbosity": "high"}  # can toggle to "low", "medium" for different behaviors

        self.c0 = Cell(x=1.0, y=1.0, z=1.0, velocity=[1.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
        self.c_xp = Cell(x=2.0, y=1.0, z=1.0, velocity=[2.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
        self.c_xm = Cell(x=0.0, y=1.0, z=1.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
        self.c_yp = Cell(x=1.0, y=2.0, z=1.0, velocity=[0.0, 1.0, 0.0], pressure=0.0, fluid_mask=True)
        self.c_ym = Cell(x=1.0, y=0.0, z=1.0, velocity=[0.0, -1.0, 0.0], pressure=0.0, fluid_mask=True)
        self.c_zp = Cell(x=1.0, y=1.0, z=2.0, velocity=[0.0, 0.0, 1.0], pressure=0.0, fluid_mask=True)
        self.c_zm = Cell(x=1.0, y=1.0, z=0.0, velocity=[0.0, 0.0, -1.0], pressure=0.0, fluid_mask=True)

        self.grid = [self.c0, self.c_xp, self.c_xm, self.c_yp, self.c_ym, self.c_zp, self.c_zm]

    def test_divergence_of_fluid_cell(self):
        div = compute_divergence(self.c0, self.grid, self.spacing)
        self.assertAlmostEqual(div, 1.0, places=5)

    def test_non_fluid_cell_returns_zero(self):
        ghost = Cell(x=5.0, y=5.0, z=5.0, velocity=[1.0, 1.0, 1.0], pressure=0.0, fluid_mask=False)
        result = compute_divergence(ghost, [ghost], self.spacing)
        self.assertEqual(result, 0.0)

    def test_missing_neighbors_use_self_velocity(self):
        isolated = Cell(x=10.0, y=10.0, z=10.0, velocity=[0.0, 0.0, 0.0], pressure=0.0, fluid_mask=True)
        div = compute_divergence(isolated, [isolated], self.spacing)
        self.assertEqual(div, 0.0)

    def test_compute_divergence_stats_summary(self):
        stats = compute_divergence_stats(self.grid, self.spacing, label="test_stage", config=self.config)
        self.assertGreater(stats["max"], 0.0)
        self.assertGreater(stats["mean"], 0.0)
        self.assertEqual(stats["count"], 7)

    def test_compute_divergence_stats_empty_grid(self):
        stats = compute_divergence_stats([], self.spacing, config=self.config)
        self.assertEqual(stats["max"], 0.0)
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["count"], 0)

    def test_compute_divergence_stats_writes_log_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = compute_divergence_stats(
                self.grid,
                self.spacing,
                label="test_write",
                step_index=42,
                output_folder=tmpdir,
                config=self.config
            )
            log_path = os.path.join(tmpdir, "divergence_log.txt")
            self.assertTrue(os.path.exists(log_path))
            with open(log_path, "r") as f:
                content = f.read()
            self.assertIn("Step 0042", content)
            self.assertIn("Stage: test_write", content)

    def test_dump_divergence_map_structure(self):
        result = dump_divergence_map(self.grid, self.spacing, config=self.config)
        self.assertEqual(len(result), 7)
        for entry in result:
            self.assertIn("x", entry)
            self.assertIn("y", entry)
            self.assertIn("z", entry)
            self.assertIn("divergence", entry)

    def test_dump_divergence_map_file_write(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            path = f.name
        try:
            data = dump_divergence_map(self.grid, self.spacing, path=path, config=self.config)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(len(data) > 0)
        finally:
            os.remove(path)

if __name__ == "__main__":
    unittest.main()



