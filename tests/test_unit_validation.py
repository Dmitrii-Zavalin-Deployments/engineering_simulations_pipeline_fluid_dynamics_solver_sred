import json
import unittest
from jsonschema import validate, ValidationError

class TestInputValidation(unittest.TestCase):
    def setUp(self):
        """Load input JSON"""
        with open("data/testing-input-output/boundary_conditions.json") as f:
            self.input_data = json.load(f)

    def test_json_schema(self):
        """Ensure input file follows defined JSON schema"""
        schema = {
            "type": "object",
            "properties": {
                "inlet_boundary": {"type": "object"},
                "outlet_boundary": {"type": "object"},
                "wall_boundary": {"type": "object"},
                "simulation_settings": {"type": "object"}
            },
            "required": ["inlet_boundary", "outlet_boundary", "wall_boundary", "simulation_settings"]
        }
        try:
            validate(instance=self.input_data, schema=schema)
        except ValidationError:
            self.fail("JSON schema validation failed!")

    def test_numerical_validity(self):
        """Ensure fluid properties are within physical constraints"""
        assert 0 <= self.input_data["inlet_boundary"]["velocity"][0] <= 50, "Velocity out of range!"
        assert 101325 <= self.input_data["inlet_boundary"]["pressure"], "Pressure unrealistic!"
        assert 0.0008 <= self.input_data["inlet_boundary"]["fluid_properties"]["viscosity"] <= 0.0015, "Viscosity invalid!"

if __name__ == "__main__":
    unittest.main()



