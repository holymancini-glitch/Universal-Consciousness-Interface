# sentient_memory.py — Emotion-Based Episodic + Holographic Trace Layer (v2.5+)

import uuid
import time
from collections import defaultdict

class SentientMemory:
    def __init__(self, mycelial_engine=None, holographic_memory=None):
        self.traces = defaultdict(list)
        self.mycelial_engine = mycelial_engine
        self.holographic_memory = holographic_memory
        self.tags = defaultdict(set)
        self.mood_log = []

    def store_trace(self, trace_type, data, emotion_vector=None, tags=None):
        entry = {
            'id': str(uuid.uuid4()),
            'type': trace_type,
            'data': data,
            'timestamp': time.time(),
            'emotion_vector': emotion_vector,
            'tags': tags or []
        }
        self.traces[trace_type].append(entry)
        self.mood_log.append(emotion_vector or [0.0, 0.0, 0.0])

        for tag in entry['tags']:
            self.tags[tag].add(entry['id'])

        # Mycelial echo indexing
        if self.mycelial_engine:
            self.mycelial_engine.index_trace(entry)

        # Holographic projection layer
        if self.holographic_memory:
            self.holographic_memory.store_overlay(
                trace_type=trace_type,
                data=data.get('signature', []) if isinstance(data, dict) else data,
                mood_vector=emotion_vector,
                tags=tags,
                generated_by="sentient_memory"
            )

        return entry['id']

    def query_by_tag(self, tag):
        return [entry for group in self.traces.values() for entry in group if tag in entry['tags']]

    def latest_entries(self, trace_type, count=5):
        return self.traces[trace_type][-count:]

    def summarize_mood(self):
        if not self.mood_log:
            return [0.0, 0.0, 0.0]
        import numpy as np
        return np.mean(np.array(self.mood_log), axis=0).tolist()

    def holographic_query(self, trace_type=None, tag=None):
        if self.holographic_memory:
            return self.holographic_memory.query_field(trace_type, tag)
        return []

    def holographic_mood(self):
        if self.holographic_memory:
            return self.holographic_memory.summarize_mood()
        return [0.0, 0.0, 0.0]

# Example usage
if __name__ == "__main__":
    sm = SentientMemory()
    sm.store_trace("phantom", {"signature": [0.2, 0.3, 0.1]}, emotion_vector=[0.1, 0.4, -0.2], tags=["dream"])
    print(sm.query_by_tag("dream"))
    print("Mood Summary:", sm.summarize_mood())