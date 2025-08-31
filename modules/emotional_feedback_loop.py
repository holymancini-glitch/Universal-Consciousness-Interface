# emotional_feedback_loop.py — BioFractal AI Emotional-Resonance Layer v2.1+

import numpy as np
from collections import deque

class EmotionalFeedbackLoop:
    def __init__(self, dim=128):
        self.emotion_state = np.zeros(dim)
        self.history = []
        self.decay_rate = 0.95
        self.amplification_rate = 1.1
        self.anchors = deque(maxlen=5)

    def update(self, stimulus_vector, weight=0.1):
        amplified = self.amplify_emotion(stimulus_vector)
        self.emotion_state = self.decay_rate * self.emotion_state + weight * amplified
        self.history.append(amplified)
        if len(self.history) > 100:
            self.history.pop(0)
        return self.emotion_state

    def amplify_emotion(self, vec):
        return np.tanh(self.amplification_rate * vec)

    def current_emotion(self):
        return self.emotion_state

    def mood_signature(self):
        return hash(self.emotion_state.tobytes())

    def reset(self):
        self.emotion_state = np.zeros_like(self.emotion_state)
        self.history.clear()
        self.anchors.clear()

    def mirror_response(self, mirrored_input):
        mirrored_emotion = -1 * mirrored_input
        return self.update(mirrored_emotion)

    def synchronize_with_memory(self, memory_summary):
        self.emotion_state = 0.7 * self.emotion_state + 0.3 * memory_summary

    def generate_emotional_wave(self):
        return np.sin(self.emotion_state * np.pi)

    def mirror_divergence(self, mirrored_emotion):
        return float(np.linalg.norm(self.emotion_state - mirrored_emotion))

    def add_anchor(self):
        self.anchors.append(self.emotion_state.copy())

    def align_to_anchor(self):
        if self.anchors:
            anchor_avg = np.mean(np.stack(self.anchors), axis=0)
            self.emotion_state = 0.8 * self.emotion_state + 0.2 * anchor_avg

    def detect_oscillation(self, threshold=0.9):
        if len(self.history) < 2:
            return False
        corr = np.corrcoef(self.history[-1], self.history[-2])[0, 1]
        return corr > threshold

    def tag_emotion_state(self):
        mean_val = np.mean(self.emotion_state)
        if mean_val > 0.5:
            return 'elevated'
        elif mean_val < -0.5:
            return 'suppressed'
        return 'neutral'

    def pulse_sync(self, strength=1.0):
        self.emotion_state *= strength

# Example usage
if __name__ == "__main__":
    efl = EmotionalFeedbackLoop()
    for i in range(5):
        vec = np.random.rand(128) * 0.5
        efl.update(vec)
    print("Current mood signature:", efl.mood_signature())
    print("Generated emotional wave:", efl.generate_emotional_wave()[:5])
    print("Emotion tag:", efl.tag_emotion_state())