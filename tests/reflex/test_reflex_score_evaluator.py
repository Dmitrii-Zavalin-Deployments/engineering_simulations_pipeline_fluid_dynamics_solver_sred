# Filename: tests/reflex/test_reflex_score_evaluator.py

import pytest
from metrics import reflex_score_evaluator

@pytest.mark.parametrize("summary_text,expected_score_tags", [
    (
        # Case: Mutation due to ghost influence
        """[🔄 Step 0 Summary]
• Fluid–ghost adjacents: 2
• Ghost influence applied: True
• Pressure mutated: True
""",
        ["mutation_detected"]
    ),
    (
        # Case: Ghost adjacency but fields matched → mutation suppressed
        """[🔄 Step 1 Summary]
• Fluid–ghost adjacents: 2
• Ghost influence applied: False
• Pressure mutated: False
• Influence suppressed: 2
""",
        ["ghost_adjacency_no_mutation"]
    ),
    (
        # Case: No ghost adjacency
        """[🔄 Step 2 Summary]
• Fluid–ghost adjacents: 0
• Pressure mutated: False
""",
        []
    )
])
def test_reflex_score_tags(tmp_path, summary_text, expected_score_tags):
    path = tmp_path / "step_summary.txt"
    path.write_text(summary_text)

    score_data = reflex_score_evaluator.evaluate_reflex_score(str(path))
    tags = score_data.get("score_tags", [])

    for tag in expected_score_tags:
        assert tag in tags

    # Ensure suppression or adjacency is not mis-tagged
    extraneous_tags = set(tags) - set(expected_score_tags)
    assert not extraneous_tags



