# tests/helpers/run_all_integrations.py

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

tests = [
    {
        "module": "config_validator",
        "path": "src/step1_input_data_parsing/config_validator.py",
        "expected": "tests/test_models/config_validator_output.json"
    },
    {
        "module": "input_reader",
        "path": "src/step1_input_data_parsing/input_reader.py",
        "expected": "tests/test_models/input_reader_output.json"
    }
]

input_file = "tests/test_models/test_model_input.json"
compare_script = "tests/helpers/compare_json.py"

def integration_tests_runner(module_path: str, input_path: str, output_path: str):
    try:
        with open(input_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load input file: {e}")
        sys.exit(1)

    try:
        module_path = Path(module_path).resolve()
        module_name = module_path.stem
        spec = importlib.util.spec_from_file_location(module_name, str(module_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.validate_config(config)
    except Exception as e:
        print(f"❌ Validation failed in {module_name}: {e}")
        sys.exit(1)

    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"❌ Failed to write output file: {e}")
        sys.exit(1)

    print("✅ Validation completed successfully.")

def run_test(module_name, module_path, expected_path):
    temp_output = f"tests/test_models/{module_name}_cli_output.json"

    print(f"\n--- Running test: {module_name}.py ---")

    if not Path(input_file).is_file() or not Path(expected_path).is_file():
        print(f"❌ Fatal Error: Missing input or expected output for {module_name}")
        sys.exit(1)

    integration_tests_runner(module_path, input_file, temp_output)

    subprocess.run([
        "python3", compare_script,
        expected_path,
        temp_output
    ], check=True)

    print(f"✅ Integration test completed for {module_name}.py")

def main():
    for test in tests:
        run_test(test["module"], test["path"], test["expected"])

if __name__ == "__main__":
    main()



