# validate_thresholds.py

"""
CLI Tool: Validates `test_thresholds.json` against a schema and audits fallback risk.
Also generates `threshold_report.json` in data/testing-input-output/.

Usage:
    python validate_thresholds.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from jsonschema import validate, ValidationError

THRESHOLD_PATH = os.path.join("src", "test_thresholds.json")
SCHEMA_PATH = os.path.join("schema", "thresholds.schema.json")
REPORT_PATH = os.path.join("data", "testing-input-output", "threshold_report.json")

FALLBACK_KEYS = {
    "volatility_tests": {
        "warning_threshold": -1.0,
        "max_slope_per_step": -1.0
    },
    "cfl_tests": {
        "max_cfl_stable": -1.0
    },
    "projection_effectiveness": {
        "minimum_reduction_percent": -1.0
    }
}

def load_json(path, label):
    if not os.path.isfile(path):
        print(f"❌ {label} file not found at: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def check_schema(thresholds, schema):
    try:
        validate(instance=thresholds, schema=schema)
        print("✅ Schema validation passed.")
        return "valid"
    except ValidationError as e:
        print(f"❌ Schema validation failed: {e.message}")
        return "invalid"

def check_fallbacks(thresholds):
    fallback_messages = []
    for section, keys in FALLBACK_KEYS.items():
        config_section = thresholds.get(section, {})
        for key, fallback_value in keys.items():
            actual = config_section.get(key, fallback_value)
            if actual == fallback_value:
                warning = f"{section}.{key}: used fallback {fallback_value}"
                warnings.warn(f"[FALLBACK] {warning}")
                fallback_messages.append(warning)
    if not fallback_messages:
        print("✅ No fallback values detected.")
    else:
        print("⚠️ Fallback values were used. Review above.")
    return fallback_messages

def write_report(thresholds, schema_status, fallback_messages):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "source_file": THRESHOLD_PATH,
        "schema_status": schema_status,
        "fallback_warnings": fallback_messages,
        "thresholds": thresholds
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Threshold report saved to: {REPORT_PATH}")

def main():
    print("🔍 Loading threshold config and schema...")
    thresholds = load_json(THRESHOLD_PATH, "Thresholds")
    schema = load_json(SCHEMA_PATH, "Schema")

    print("🔍 Running schema validation...")
    schema_status = check_schema(thresholds, schema)

    print("🔍 Auditing fallback usage...")
    fallback_messages = check_fallbacks(thresholds)

    print("📝 Writing threshold report...")
    write_report(thresholds, schema_status, fallback_messages)

    print("🏁 Validation complete.")

if __name__ == "__main__":
    main()



