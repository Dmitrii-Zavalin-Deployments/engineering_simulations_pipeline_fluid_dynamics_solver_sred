# tests/__init__.py

import os
import sys

# ✅ Add current directory and project root to sys.path for import resolution
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))

for path in [TEST_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)



