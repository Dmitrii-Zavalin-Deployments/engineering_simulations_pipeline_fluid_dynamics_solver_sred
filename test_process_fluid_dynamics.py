import unittest
import json
import os
from process_fluid_dynamics import load_boundary_conditions, process_fluid_dynamics, save_output

class TestFluidDynamicsCalculations(unittest.TestCase):

    def setUp(self):
        """Prepare sample input data for testing"""
        self.test_input_file = "testing-input-output/test_boundary_conditions.json"
        self.test_output_file = "testing-input-output/test_fluid_flow_parameters.json"
        
        # Sample boundary conditions
        self.sample_conditions = {
            "velocity_x": 1.5,
            "velocity_y": -0.5,
            "velocity_z": 2.0,
            "pressure": 101325  # Atmospheric pressure in Pascals
        }
        
        # Write sample JSON input
        with open(self.test_input_file, 'w') as f:
            json.dump(self.sample_conditions, f, indent=4)

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

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_input_file):
            os.remove(self.test_input_file)
        if os.path.exists(self.test_output_file):
            os.remove(self.test_output_file)

if __name__ == '__main__':
    unittest.main()
