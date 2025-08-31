# holographic_memory.py — Memory Overlays & Field-Based Recall (v1.5+)

import numpy as np
import uuid
from collections import defaultdict

class HolographicMemory:
    def __init__(self):
        self.memory_fields = defaultdict(list)  # key: tag/type, value: list of overlays
        self.mood_tracker = []
        self.fade_rate = 0.01  # default temporal fade

    def store_overlay(self, trace_type, data, mood_vector=None, tags=None, generated_by=None):
        overlay = {
            'id': str(uuid.uuid4()),
            'type': trace_type,
            'data': data,
            'tags': tags or [],
            'mood_vector': mood_vector,
            'generated_by': generated_by,
            'decay': 1.0,
            'timestamp': self._current_time()
        }
        self.memory_fields[trace_type].append(overlay)
        if mood_vector:
            self.mood_tracker.append(mood_vector)
        return overlay['id']

    def decay_all(self):
        for overlays in self.memory_fields.values():
            for o in overlays:
                time_elapsed = self._current_time() - o.get('timestamp', 0)
                o['decay'] = max(0.0, 1.0 - self.fade_rate * time_elapsed)

    def fuse_similar(self, trace_type, threshold=0.9):
        overlays = self.memory_fields[trace_type]
        fused = []
        skip = set()
        for i, a in enumerate(overlays):
            if i in skip: continue
            group = [a]
            for j, b in enumerate(overlays):
                if i != j and self._similarity(a['data'], b['data']) > threshold:
                    group.append(b)
                    skip.add(j)
            if len(group) > 1:
                fused_data = np.mean([np.array(item['data']) for item in group], axis=0).tolist()
                fused.append({
                    'id': str(uuid.uuid4()),
                    'type': trace_type,
                    'data': fused_data,
                    'tags': list(set(tag for g in group for tag in g['tags'])),
                    'mood_vector': np.mean([g['mood_vector'] for g in group if g['mood_vector']], axis=0).tolist(),
                    'generated_by': group[0].get('generated_by'),
                    'decay': max(g['decay'] for g in group),
                    'timestamp': min(g['timestamp'] for g in group)
                })
            else:
                fused.append(a)
        self.memory_fields[trace_type] = fused

    def query_field(self, trace_type=None, tag=None):
        overlays = []
        if trace_type:
            overlays = self.memory_fields.get(trace_type, [])
        else:
            for items in self.memory_fields.values():
                overlays.extend(items)
        if tag:
            overlays = [o for o in overlays if tag in o['tags']]
        return sorted(overlays, key=lambda x: -x['decay'])

    def summarize_mood(self):
        if not self.mood_tracker:
            return [0.0, 0.0, 0.0]
        return np.mean(np.array(self.mood_tracker), axis=0).tolist()

    def generate_overlay(self, trace_type, seed_vector, perturbation=0.05, tags=None):
        noisy = np.tanh(seed_vector + np.random.normal(0, perturbation, size=len(seed_vector)))
        return self.store_overlay(trace_type, noisy.tolist(), tags=tags, generated_by="overlay_gen")

    def _similarity(self, a, b):
        a_vec = np.array(a)
        b_vec = np.array(b)
        dot = np.dot(a_vec, b_vec)
        norm = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
        return dot / norm if norm > 0 else 0.0

    def _current_time(self):
        import time
        return int(time.time())

# Example usage
if __name__ == "__main__":
    hm = HolographicMemory()
    id1 = hm.store_overlay("phantom", [0.1, 0.2, 0.3], mood_vector=[0.3, 0.1, -0.2], tags=["dream"])
    id2 = hm.store_overlay("phantom", [0.11, 0.21, 0.29], mood_vector=[0.2, 0.1, -0.1], tags=["dream"])
    hm.fuse_similar("phantom")
    hm.decay_all()
    print(hm.query_field("phantom"))
    print("Mood summary:", hm.summarize_mood())