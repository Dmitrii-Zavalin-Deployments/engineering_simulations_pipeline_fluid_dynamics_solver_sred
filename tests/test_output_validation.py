import json
import unittest
import numpy as np
import os

class TestOutputValidation(unittest.TestCase):
    def setUp(self):
        """Load output JSON"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, '..', 'data', 'testing-input-output', 'simulation_results.json')

        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Expected simulation results file not found at: {json_file_path}")

        with open(json_file_path) as f:
            self.output_data = json.load(f)

        # Debugging: Print keys to check if 'data_points' exists
        print("DEBUG: Output Data Keys:", self.output_data.keys())

    def test_computed_values(self):
        """Ensure computed fluid properties are valid"""
        if "data_points" not in self.output_data:
            raise KeyError("Expected 'data_points' key not found in output_data")

        for point in self.output_data["data_points"]:
            assert 0 <= point["velocity"]["components"][0] <= 50, "Velocity calculation incorrect!"
            assert point["pressure"]["value"] >= 101325, "Pressure value below realistic threshold!"
            assert 0.0008 <= point["viscosity"]["value"] <= 0.0015, "Incorrect viscosity!"

    def test_l2_norm_error(self):
        """Validate simulation results against expected benchmarks using L2 norm"""
        benchmark_velocity = [1.5]  # Adjust based on actual expected values
        
        if "data_points" not in self.output_data:
            raise KeyError("Expected 'data_points' key not found in output_data")

        computed_velocity = [point["velocity"]["components"][0] for point in self.output_data["data_points"]]

        if len(computed_velocity) > 0 and len(benchmark_velocity) != len(computed_velocity):
            if len(benchmark_velocity) == 1:
                benchmark_velocity = [benchmark_velocity[0]] * len(computed_velocity)
            else:
                raise ValueError("Benchmark velocity dimension does not match computed velocity dimension for L2 norm calculation.")

        error = np.linalg.norm(np.array(computed_velocity) - np.array(benchmark_velocity)) / np.linalg.norm(benchmark_velocity)
        assert error < 0.05, "L2 norm error too high!"

if __name__ == "__main__":
    unittest.main()



