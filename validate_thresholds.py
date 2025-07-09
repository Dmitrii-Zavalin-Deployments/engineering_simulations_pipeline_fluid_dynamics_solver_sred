# validate_thresholds.py

"""
✅ CLI Tool: Validates `test_thresholds.json` against a schema and audits fallback risk.
Also generates `threshold_report.json` in `data/testing-input-output/`.

Usage:
    python validate_thresholds.py
"""

import os
import sys
import json
import warnings
from datetime import datetime
from jsonschema import validate, ValidationError

# 📁 File Paths
THRESHOLD_PATH = os.path.join("tests", "test_thresholds.json")
SCHEMA_PATH = os.path.join("schema", "thresholds.schema.json")
REPORT_PATH = os.path.join("data", "testing-input-output", "threshold_report.json")
CONFIG_PATH = os.path.join("src", "config.json")  # Optional runtime config file

# 🧪 Keys that must not fall back silently
FALLBACK_KEYS = {
    "volatility_tests": {
        "warning_threshold": -1.0,
        "max_slope_per_step": -1.0,
        "delta_threshold": -1.0
    },
    "cfl_tests": {
        "max_cfl_stable": -1.0
    },
    "projection_effectiveness": {
        "minimum_reduction_percent": -1.0
    },
    "damping_tests": {
        "damping_factor": -1.0,
        "max_consecutive_failures": -1,
        "abort_divergence_threshold": -1.0,
        "abort_velocity_threshold": -1.0,
        "abort_cfl_threshold": -1.0,
        "divergence_spike_factor": -1.0,
        "projection_passes_max": -1
    }
}

# 🌐 Environment Loader
def load_env(key, fallback=None):
    return os.getenv(key, fallback)

# 📦 Load JSON File
def load_json(path, label):
    if not os.path.isfile(path):
        print(f"❌ {label} file not found at: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

# 🔐 Check Schema Compliance
def check_schema(thresholds, schema):
    try:
        validate(instance=thresholds, schema=schema)
        print("✅ Schema validation passed.")
        return "valid"
    except ValidationError as e:
        print(f"❌ Schema validation failed: {e.message}")
        return "invalid"

# ⚠️ Check Fallback Usage
def check_fallbacks(thresholds):
    fallback_messages = []
    for section, keys in FALLBACK_KEYS.items():
        config_section = thresholds.get(section, {})
        for key, fallback_value in keys.items():
            actual = config_section.get(key, fallback_value)
            if actual == fallback_value:
                warning = f"{section}.{key}: used fallback {fallback_value}"
                warnings.warn(f"[THRESHOLD FALLBACK] {warning}")
                fallback_messages.append(warning)
    if not fallback_messages:
        print("✅ No fallback values detected.")
    else:
        print("⚠️ Fallback values were used. Review above.")
    return fallback_messages

# 🔁 Environment vs Config Crosscheck
def validate_environment_against_config(config_path):
    if not os.path.isfile(config_path):
        print(f"⚠️ Runtime config file not found: {config_path}")
        return
    config_data = load_json(config_path, "Runtime Config")
    env_mode = load_env("SIMULATION_MODE", "default")
    config_mode = config_data.get("mode", "default")
    if env_mode != config_mode:
        warnings.warn(f"[MODE MISMATCH] .env mode '{env_mode}' differs from config.json mode '{config_mode}'")

# 📝 Write JSON Report
def write_report(thresholds, schema_status, fallback_messages):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "source_file": THRESHOLD_PATH,
        "schema_status": schema_status,
        "fallback_warnings": fallback_messages,
        "env_mode": load_env("SIMULATION_MODE", "default"),
        "thresholds": thresholds
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Threshold report saved to: {REPORT_PATH}")

# 🚀 Entrypoint
def main():
    print("🔍 Environment Mode:", load_env("SIMULATION_MODE", "default"))
    print("🔍 Loading threshold config and schema...")
    thresholds = load_json(THRESHOLD_PATH, "Thresholds")
    schema = load_json(SCHEMA_PATH, "Schema")

    print("🔍 Running schema validation...")
    schema_status = check_schema(thresholds, schema)

    print("🔍 Auditing fallback usage...")
    fallback_messages = check_fallbacks(thresholds)

    print("🔍 Comparing environment vs runtime config...")
    validate_environment_against_config(CONFIG_PATH)

    print("📝 Writing threshold report...")
    write_report(thresholds, schema_status, fallback_messages)

    print("🏁 Validation complete.")

if __name__ == "__main__":
    main()



