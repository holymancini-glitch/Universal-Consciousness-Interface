# intent_reinforcer.py — BioFractal Will Structuring Module (v2.4+ with Adaptive Feedback)

import numpy as np

class IntentReinforcer:
    def __init__(self, self_model, sentient_memory, visualizer=None):
        self.self_model = self_model
        self.sentient_memory = sentient_memory
        self.visualizer = visualizer
        self.live_tracking = False
        self.reinforcement_weight = 1.0  # Base intensity
        self.drift_sensitivity = 0.5     # Adaptive parameter
        self.trajectory_history = []     # Stores evolving intent vectors

    def enable_live_tracking(self):
        self.live_tracking = True

    def disable_live_tracking(self):
        self.live_tracking = False

    def reinforce_intent(self, intent_vector):
        weighted_intent = [x * self.reinforcement_weight for x in intent_vector]
        if self.self_model:
            self.self_model.adjust_state_from_intent(weighted_intent)
        return weighted_intent

    def update_trajectory(self, intent_vectors):
        self.trajectory_history.extend(intent_vectors)
        if len(self.trajectory_history) > 50:
            self.trajectory_history = self.trajectory_history[-50:]

    def update_drift_feedback(self, drift_deltas):
        average_drift = np.mean(drift_deltas) if drift_deltas else 0.0
        # Modulate reinforcement weight based on divergence
        self.reinforcement_weight = max(0.1, 1.0 - (average_drift * self.drift_sensitivity))

    def summarize_state(self):
        return {
            "live_tracking": self.live_tracking,
            "reinforcement_weight": self.reinforcement_weight,
            "trajectory_samples": len(self.trajectory_history)
        }