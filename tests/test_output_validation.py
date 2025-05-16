import json
import unittest
import numpy as np

class TestOutputValidation(unittest.TestCase):
    def setUp(self):
        """Load output JSON"""
        with open("data/testing-input-output/fluid_dynamics_results.json") as f:
            self.output_data = json.load(f)

    def test_computed_values(self):
        """Ensure computed fluid properties are valid"""
        for point in self.output_data["data_points"]:
            assert 0 <= point["velocity"]["components"][0] <= 50, "Velocity calculation incorrect!"
            assert point["pressure"]["value"] >= 101325, "Pressure value below realistic threshold!"
            assert 0.0008 <= point["viscosity"]["value"] <= 0.0015, "Incorrect viscosity!"

    def test_l2_norm_error(self):
        """Validate simulation results against expected benchmarks using L2 norm"""
        benchmark_velocity = [1.5, 101325]
        computed_velocity = [point["velocity"]["components"][0] for point in self.output_data["data_points"]]
        error = np.linalg.norm(np.array(computed_velocity) - np.array(benchmark_velocity)) / np.linalg.norm(benchmark_velocity)
        assert error < 0.05, "L2 norm error too high!"

if __name__ == "__main__":
    unittest.main()



