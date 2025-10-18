# src/tools/schema_validator.py

import json
import sys
import os
from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions

SCHEMA_PATH = os.path.join(
    "engineering_simulations_pipeline_fluid_dynamics_solver_sred",
    "schema",
    "fluid_simulation_input.schema.json"
)

def load_json(file_path: str) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON file: {file_path}\n{e}")
        sys.exit(1)

def validate_schema(instance: dict, schema: dict) -> None:
    try:
        Draft202012Validator(schema).validate(instance)
        print("✅ Input file is schema-valid.")
    except jsonschema_exceptions.ValidationError as ve:
        print("❌ Schema validation failed:")
        print(f"→ {ve.message}")
        if ve.path:
            print(f"→ Location: {'/'.join(map(str, ve.path))}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error during validation:\n{e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print("❌ Usage: python schema_validator.py <input_file.json>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.isfile(SCHEMA_PATH):
        print(f"❌ Schema file not found at: {SCHEMA_PATH}")
        sys.exit(1)

    schema = load_json(SCHEMA_PATH)
    instance = load_json(input_file)
    validate_schema(instance, schema)

if __name__ == "__main__":
    main()



