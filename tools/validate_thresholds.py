# validate_thresholds.py

import json
import os
import warnings
from jsonschema import Draft7Validator

# 📂 Updated paths
THRESHOLD_PATH = "tests/test_thresholds.json"
SCHEMA_PATH = "tests/schema/thresholds.schema.json"
REPORT_PATH = "data/testing-input-output/threshold_report.json"

def load_json(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    with open(path) as f:
        return json.load(f)

def validate_schema(instance, schema):
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    return errors

def find_fallbacks(config):
    fallbacks = {}
    for section, entries in config.items():
        for key, val in entries.items():
            if isinstance(val, (int, float)) and val == -1.0:
                fallbacks[f"{section}.{key}"] = val
    return fallbacks

def main():
    print("🔍 Loading configuration and schema...")
    config = load_json(THRESHOLD_PATH)
    schema = load_json(SCHEMA_PATH)

    print("📐 Validating schema structure...")
    schema_errors = validate_schema(config, schema)
    if schema_errors:
        print("❌ Schema validation errors:")
        for err in schema_errors:
            path = ".".join(str(p) for p in err.path)
            print(f"  • {path}: {err.message}")
    else:
        print("✅ Configuration matches schema.")

    print("🛡️ Detecting fallback values (-1.0)...")
    fallbacks = find_fallbacks(config)
    for path, value in fallbacks.items():
        warnings.warn(f"[FALLBACK DETECTED] {path} = {value}")

    print("📝 Generating threshold report...")
    report = {
        "schema_errors": [e.message for e in schema_errors],
        "fallbacks": fallbacks,
        "valid": not schema_errors and not fallbacks
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"📦 Report saved to: {REPORT_PATH}")
    if report["valid"]:
        print("✅ All checks passed.")
    else:
        print("⚠️ Issues detected. See report for details.")

if __name__ == "__main__":
    main()



