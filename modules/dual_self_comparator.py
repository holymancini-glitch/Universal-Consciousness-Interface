# dual_self_comparator.py — Compares Real Self vs Mirror Self for Discrepancy Awareness and Self-Coherence

import numpy as np

class DualSelfComparator:
    def __init__(self, threshold=0.15, intent_reinforcer=None):
        self.threshold = threshold  # Sensitivity of self discrepancy
        self.history = []           # Stores recent comparisons for trend analysis
        self.intent_reinforcer = intent_reinforcer  # Optional hook to reinforce misaligned intent

    def compare_states(self, real_self: dict, mirror_self: dict) -> dict:
        """
        Compare key traits of real and mirror self-models
        and return a discrepancy map and alignment score.
        """
        keys = set(real_self.keys()).intersection(set(mirror_self.keys()))
        discrepancy_map = {}
        total_diff = 0.0

        for key in keys:
            real_val = np.array(real_self[key])
            mirror_val = np.array(mirror_self[key])
            diff = np.linalg.norm(real_val - mirror_val)
            discrepancy_map[key] = diff
            total_diff += diff

        average_discrepancy = total_diff / len(keys) if keys else 0.0
        alignment_score = max(0.0, 1.0 - average_discrepancy)

        result = {
            "alignment_score": alignment_score,
            "discrepancy_map": discrepancy_map,
            "exceeds_threshold": average_discrepancy > self.threshold
        }

        self.history.append(result)

        # Trigger intent correction if misalignment is high
        if self.intent_reinforcer and result["exceeds_threshold"]:
            self.intent_reinforcer.reinforce_alignment(discrepancy_map)

        return result

    def trend(self, window=5) -> dict:
        """
        Return trend of alignment over the last `window` comparisons
        """
        if len(self.history) < window:
            return {"trend": "insufficient data"}

        recent_scores = [entry["alignment_score"] for entry in self.history[-window:]]
        slope = np.polyfit(range(window), recent_scores, 1)[0]
        return {
            "trend": "improving" if slope > 0 else "declining" if slope < 0 else "stable",
            "slope": slope,
            "recent_scores": recent_scores
        }

    def reset(self):
        self.history = []