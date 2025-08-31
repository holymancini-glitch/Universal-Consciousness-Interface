# mirror_world.py — Generative Reflective Simulation Environment v2.1+

import random
import hashlib

class MirrorWorld:
    def __init__(self, latent_space=None, sentient_memory=None, emotional_feedback=None):
        self.latent_space = latent_space
        self.sentient_memory = sentient_memory
        self.emotional_feedback = emotional_feedback
        self.reflections = []

    def mirror_event(self, event):
        mirrored = {
            'type': f"mirror_{event['type']}",
            'content': self._mirror_content(event.get('content', {})),
            'origin_timestamp': event.get('timestamp'),
            'emotional_reflection': self._reflect_emotion(event),
            'mirror_hash': self._generate_hash(event)
        }
        self.reflections.append(mirrored)
        return mirrored

    def _mirror_content(self, content):
        # Simple inversion/mirroring logic for demo
        return {k: self._invert_value(v) for k, v in content.items()}

    def _invert_value(self, v):
        if isinstance(v, (int, float)):
            return -v
        elif isinstance(v, str):
            return v[::-1]  # reverse string
        elif isinstance(v, list):
            return v[::-1]
        return v

    def _generate_hash(self, event):
        event_str = str(event)
        return hashlib.sha256(event_str.encode()).hexdigest()

    def _reflect_emotion(self, event):
        if self.emotional_feedback and 'emotion_vector' in event:
            inverted = [-1 * e for e in event['emotion_vector']]
            return inverted
        return None

    def generate_mirror_simulation(self, num_frames=3):
        sim = []
        for i in range(num_frames):
            frame = {
                'sim_time': i,
                'phantom_signature': random.random(),
                'mirrored_latent': self.latent_space.generate_signature() if self.latent_space else None
            }
            sim.append(frame)
        return sim

    def receive(self, event):
        mirrored = self.mirror_event(event)
        if self.sentient_memory:
            self.sentient_memory.store_trace("mirror_event", mirrored)
        return mirrored

# Example usage
if __name__ == "__main__":
    mw = MirrorWorld()
    event = {'type': 'input', 'content': {'msg': 'hello'}, 'timestamp': 42, 'emotion_vector': [0.1, -0.3, 0.5]}
    mirrored = mw.mirror_event(event)
    print(mirrored)