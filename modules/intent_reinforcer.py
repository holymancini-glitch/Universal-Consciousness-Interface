# intent_reinforcer.py — Intent Strengthening Module with Live Tracking (v2.3+)

import numpy as np

class IntentReinforcer:
    def __init__(self, self_model=None, sentient_memory=None, visualizer=None):
        self.self_model = self_model
        self.sentient_memory = sentient_memory
        self.visualizer = visualizer
        self.intent_trace = []
        self.live_tracking = False  # 🔁 Toggle for dashboard/live mode

    def enable_live_tracking(self):
        self.live_tracking = True

    def disable_live_tracking(self):
        self.live_tracking = False

    def reinforce(self, intent_vector, context=None):
        reinforced = np.tanh(intent_vector * 1.2)
        intent_event = {
            'intent_vector': reinforced.tolist(),
            'context': context or {},
            'mood_signature': self._extract_mood(intent_vector)
        }

        self.intent_trace.append(intent_event)

        if self.sentient_memory:
            self.sentient_memory.store_trace("reinforced_intent", intent_event)

        if self.self_model:
            self.self_model.apply_intention(intent_vector)

        if self.live_tracking and self.visualizer:
            self.visualizer.plot_emotional_wave([intent_event['mood_signature']])
            self.visualizer.plot_latent_projection([reinforced.tolist()])

        return reinforced

    def _extract_mood(self, vec):
        return [float(np.mean(vec)), float(np.max(vec)), float(np.min(vec))]

    def trace_log(self):
        return self.intent_trace

    def dashboard_toggle(self, enable: bool):
        """
        Enable or disable live dashboard tracking.
        """
        if enable:
            self.enable_live_tracking()
        else:
            self.disable_live_tracking()

# Example usage
if __name__ == "__main__":
    ir = IntentReinforcer()
    ir.dashboard_toggle(True)
    sample = np.array([0.4, 0.6, 0.2])
    result = ir.reinforce(sample)
    print("Reinforced:", result)
