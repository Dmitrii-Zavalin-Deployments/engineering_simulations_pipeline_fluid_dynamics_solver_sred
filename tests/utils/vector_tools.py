# tests/utils/vector_tools.py
# 📐 Vector diagnostic utilities for fluid snapshot validation

import math

def vector_angle(v1, v2):
    """Returns the angle (in radians) between two 3D vectors v1 and v2."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a ** 2 for a in v1))
    norm2 = math.sqrt(sum(b ** 2 for b in v2))
    return math.acos(dot / (norm1 * norm2 + 1e-8))  # radians



