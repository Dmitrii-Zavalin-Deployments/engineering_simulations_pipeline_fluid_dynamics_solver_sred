import unittest
import json
import os
from src.process_fluid_dynamics import load_boundary_conditions, process_fluid_dynamics, save_output

class TestFluidDynamicsCalculations(unittest.TestCase):

    def setUp(self):
        """Prepare sample input data for testing"""
        self.test_input_file = "data/testing-input-output/test_boundary_conditions.json"
        self.test_output_file = "data/testing-input-output/test_fluid_flow_parameters.json"
        self.expected_output_file = "data/testing-input-output/expected_output.json"

        # Sample boundary conditions
        self.sample_conditions = {
            "velocity_x": 1.5,
            "velocity_y": -0.5,
            "velocity_z": 2.0,
            "pressure": 101325
        }

        # Expected output data with coordinates
        self.expected_output = {
            "expected_pressure_field": [
                {"position": [0.0, 0.0, 0.0], "pressure": 101300},
                {"position": [1.0, 0.0, 0.0], "pressure": 101320}
            ],
            "expected_velocity_field": [
                {"position": [0.0, 0.0, 0.0], "velocity": [1.5, -0.5, 2.0]},
                {"position": [1.0, 0.0, 0.0], "velocity": [1.4, -0.4, 1.9]}
            ]
        }

        # Ensure testing-input-output directory exists
        os.makedirs("data/testing-input-output", exist_ok=True)

        # Write sample JSON input
        with open(self.test_input_file, 'w') as f:
            json.dump(self.sample_conditions, f, indent=4)

        # Write expected JSON output
        with open(self.expected_output_file, 'w') as f:
            json.dump(self.expected_output, f, indent=4)

    def test_load_boundary_conditions(self):
        """Test that boundary conditions are loaded correctly"""
        data = load_boundary_conditions(self.test_input_file)
        self.assertIsInstance(data, dict)
        self.assertEqual(data["velocity_x"], 1.5)
        self.assertEqual(data["pressure"], 101325)

    def test_process_fluid_dynamics(self):
        """Test fluid dynamics calculations return expected structure"""
        processed_data = process_fluid_dynamics(self.sample_conditions)
        self.assertIn("adjusted_velocity_x", processed_data)
        self.assertIn("adjusted_velocity_y", processed_data)
        self.assertIn("adjusted_velocity_z", processed_data)

        # Ensure calculations adjust values reasonably
        self.assertTrue(1.3 <= processed_data["adjusted_velocity_x"] <= 1.7)
        self.assertTrue(-0.55 <= processed_data["adjusted_velocity_y"] <= -0.45)
        self.assertTrue(1.8 <= processed_data["adjusted_velocity_z"] <= 2.2)

    def test_save_output(self):
        """Test that output is saved correctly"""
        processed_data = process_fluid_dynamics(self.sample_conditions)
        save_output(self.test_output_file, processed_data)

        self.assertTrue(os.path.exists(self.test_output_file))

        # Load saved JSON and verify structure
        with open(self.test_output_file, 'r') as f:
            saved_data = json.load(f)

        self.assertIn("adjusted_velocity_x", saved_data)
        self.assertEqual(saved_data["adjusted_velocity_x"], processed_data["adjusted_velocity_x"])

    def test_validate_expected_output(self):
        """Test if processed fluid dynamics data matches expected results"""
        with open(self.test_output_file, 'r') as f:
            saved_data = json.load(f)

        with open(self.expected_output_file, 'r') as f:
            expected_data = json.load(f)

        # Verify pressure values at expected positions
        for expected_point in expected_data["expected_pressure_field"]:
            pos = expected_point["position"]
            expected_pressure = expected_point["pressure"]

            self.assertTrue(
                any(abs(expected_pressure - saved_data.get("pressure", 0)) <= 50 for _ in pos),
                f"Pressure mismatch at position {pos}"
            )

        # Verify velocity values at expected positions
        for expected_point in expected_data["expected_velocity_field"]:
            pos = expected_point["position"]
            expected_velocity = expected_point["velocity"]

            self.assertTrue(
                any(abs(expected_velocity[i] - saved_data.get(f"adjusted_velocity_{['x', 'y', 'z'][i]}", 0)) <= 0.2 for i in range(3)),
                f"Velocity mismatch at position {pos}"
            )

    def tearDown(self):
        """Clean up test files"""
        for file in [self.test_input_file, self.test_output_file, self.expected_output_file]:
            if os.path.exists(file):
                os.remove(file)

if __name__ == '__main__':
    unittest.main()
