# src/ci_score_report.py
# 📊 CI Reflex Score Report — evaluates reflex integrity from step_summary.txt
# for audit dashboards and CI gating

import os
import sys

# ✅ Centralized debug flag for GitHub Actions logging
debug = True

# Ensure src/ is available on import path
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.ci.reflex_log_score import score_combined

SUMMARY_PATH = os.environ.get(
    "CI_SUMMARY_PATH",
    "data/testing-input-output/navier_stokes_output/step_summary.txt"
)


if __name__ == "__main__":
    if not os.path.exists(SUMMARY_PATH):
        print(f"❌ Summary file not found → {SUMMARY_PATH}")
    else:
        with open(SUMMARY_PATH, "r") as f:
            log_text = f.read()

        scores = score_combined(log_text, SUMMARY_PATH)

        if debug:
            print(f"[DEBUG] scores: {scores}")

        print("📊 CI Reflex Scoring Results:")
        print(
            f"↪️ Matched markers: "
            f"{scores['ci_log_score']['markers_matched']}"
        )
        print(
            f"↪️ Marker score: "
            f"{scores['ci_log_score']['reflex_score']}"
        )
        print(
            f"↪️ Summary score: "
            f"{scores['summary_score']['average_score']:.2f} (avg)"
        )

        # ✅ Patch: threshold awareness for audit visibility
        if scores["summary_score"]["average_score"] < 0.5:
            print("⚠️ Reflex score below expected threshold.")

        # ✅ Optional export
        # with open("reflex_ci_scores.json", "w") as out:
        #     json.dump(scores, out, indent=2)



