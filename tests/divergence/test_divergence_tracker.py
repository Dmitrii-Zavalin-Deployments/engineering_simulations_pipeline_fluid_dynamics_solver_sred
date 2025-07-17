# Filename: tests/divergence/test_divergence_tracker.py

import pytest
from divergence_tracker import extract_labeled_divergence

@pytest.mark.parametrize("log_text,expected_labels", [
    (
        # Case: all divergence stages present
        """Divergence stats (pre-pressure):
   Max divergence: 2.100000e-01
   Mean divergence: 1.550000e-01
   Cells evaluated: 4
Divergence stats (post-pressure):
   Max divergence: 1.800000e-01
   Mean divergence: 1.200000e-01
   Cells evaluated: 4
Divergence stats (post-velocity):
   Max divergence: 0.000000e+00
   Mean divergence: 0.000000e+00
   Cells evaluated: 4
""",
        {
            "pre-pressure": 2.1e-01,
            "post-pressure": 1.8e-01,
            "post-velocity": 0.0
        }
    ),
    (
        # Case: missing velocity projection
        """Divergence stats (pre-pressure):
   Max divergence: 1.000000e-01
Divergence stats (post-pressure):
   Max divergence: 5.000000e-02
""",
        {
            "pre-pressure": 1.0e-01,
            "post-pressure": 5.0e-02,
            "post-velocity": None
        }
    ),
])
def test_extract_labeled_divergence(log_text, expected_labels):
    result = extract_labeled_divergence(log_text)
    for label, expected in expected_labels.items():
        if expected is None:
            assert label not in result or result[label] is None
        else:
            assert abs(result[label] - expected) < 1e-6



