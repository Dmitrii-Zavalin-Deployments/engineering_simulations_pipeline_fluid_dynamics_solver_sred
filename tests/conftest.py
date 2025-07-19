# tests/conftest.py
# 🧪 Pytest configuration — ensures src/ is importable across all tests

import sys
import os

# Add the top-level src/ directory to the Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)



