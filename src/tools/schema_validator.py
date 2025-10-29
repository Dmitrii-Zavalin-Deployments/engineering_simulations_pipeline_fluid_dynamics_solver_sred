# src/tools/schema_validator.py
# 📐 Schema Validator — validates simulation input against JSON schema and reflex-aware structural rules
# 📌 This module enforces schema compliance and reflex-critical structure for boundary and ghost logic.

import json
import sys
import os
from jsonschema import Draft202012Validator, exceptions as jsonschema_exceptions

# ✅ Centralized debug flag for GitHub Actions logging
debug = False

SCHEMA_PATH = "./schema/fluid_simulation_input.schema.json"

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
        if debug:
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

    # ✅ Additional manual validation for boundary_conditions structure
    boundary_conditions = instance.get("boundary_conditions", [])
    if not isinstance(boundary_conditions, list):
        print("❌ boundary_conditions must be a list.")
        sys.exit(1)

    for i, bc in enumerate(boundary_conditions):
        if not isinstance(bc, dict):
            print(f"❌ boundary_conditions[{i}] must be a dictionary.")
            sys.exit(1)
        if "apply_to" not in bc:
            print(f"❌ boundary_conditions[{i}] missing required key: 'apply_to'.")
            sys.exit(1)

    # ✅ Optional manual validation for ghost_rules structure
    ghost_rules = instance.get("ghost_rules")
    if ghost_rules:
        if not isinstance(ghost_rules, dict):
            print("❌ ghost_rules must be a dictionary.")
            sys.exit(1)
        for key in ["boundary_faces", "default_type", "face_types"]:
            if key not in ghost_rules:
                print(f"❌ ghost_rules missing required key: {key}")
                sys.exit(1)

    if debug:
        print("✅ Manual structure validation passed.")

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



