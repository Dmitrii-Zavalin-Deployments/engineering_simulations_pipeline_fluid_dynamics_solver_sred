import json
import unittest
import numpy as np
import os # Import the os module for robust path handling

class TestOutputValidation(unittest.TestCase):
    def setUp(self):
        """Load output JSON"""
        # Get the directory of the current test file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the path to the JSON file, changing the filename
        # This assumes your 'data' directory is one level up from 'tests'
        json_file_path = os.path.join(current_dir, '..', 'data', 'testing-input-output', 'simulation_results.json')
        
        # Optional: Add a check to confirm the file exists before attempting to open
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"Expected simulation results file not found at: {json_file_path}")

        with open(json_file_path) as f:
            self.output_data = json.load(f)

    def test_computed_values(self):
        """Ensure computed fluid properties are valid"""
        for point in self.output_data["data_points"]:
            assert 0 <= point["velocity"]["components"][0] <= 50, "Velocity calculation incorrect!"
            assert point["pressure"]["value"] >= 101325, "Pressure value below realistic threshold!"
            assert 0.0008 <= point["viscosity"]["value"] <= 0.0015, "Incorrect viscosity!"

    def test_l2_norm_error(self):
        """Validate simulation results against expected benchmarks using L2 norm"""
        # This line might need adjustment based on what 'benchmark_velocity' truly represents
        # If your 'simulation_results.json' only contains one data point for velocity,
        # benchmark_velocity should ideally have one element for comparison.
        # Assuming for now that computed_velocity will have multiple elements.
        benchmark_velocity = [1.5] # Adjust based on the actual expected benchmark for velocity components
        computed_velocity = [point["velocity"]["components"][0] for point in self.output_data["data_points"]]

        # Ensure benchmark_velocity matches the dimension of computed_velocity if possible,
        # or handle cases where they differ (e.g., if benchmark is a single expected value)
        if len(computed_velocity) > 0 and len(benchmark_velocity) != len(computed_velocity):
            # If benchmark_velocity is meant to be a single value to compare against all computed,
            # replicate it for the L2 norm calculation.
            if len(benchmark_velocity) == 1:
                benchmark_velocity = [benchmark_velocity[0]] * len(computed_velocity)
            else:
                # Handle error if dimensions truly don't match for L2 norm
                raise ValueError("Benchmark velocity dimension does not match computed velocity dimension for L2 norm calculation.")

        error = np.linalg.norm(np.array(computed_velocity) - np.array(benchmark_velocity)) / np.linalg.norm(benchmark_velocity)
        assert error < 0.05, "L2 norm error too high!"

if __name__ == "__main__":
    unittest.main()



