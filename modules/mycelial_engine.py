# mycelial_engine.py — Distributed Resonance & Memory Field v1.0

import numpy as np
from collections import defaultdict, deque

class MycelialEngine:
    def __init__(self, dim=128):
        self.nodes = defaultdict(list)
        self.decay_rate = 0.995
        self.signal_threshold = 0.2
        self.trail_maxlen = 50
        self.trails = defaultdict(lambda: deque(maxlen=self.trail_maxlen))

    def grow_path(self, tag, signal_vector):
        norm = np.linalg.norm(signal_vector)
        if norm < self.signal_threshold:
            return
        self.nodes[tag].append(signal_vector)
        self.trails[tag].append(signal_vector)

    def decay(self):
        for tag in self.nodes:
            self.nodes[tag] = [vec * self.decay_rate for vec in self.nodes[tag] if np.linalg.norm(vec) > 0.05]

    def spore_seed(self, tag, seed_vector):
        self.nodes[tag] = [seed_vector]
        self.trails[tag].clear()
        self.trails[tag].append(seed_vector)

    def echo_query(self, tag):
        if tag not in self.trails or not self.trails[tag]:
            return np.zeros(128)
        echo = np.mean(np.stack(self.trails[tag]), axis=0)
        return echo / (np.linalg.norm(echo) + 1e-6)

    def cross_connect(self, tag1, tag2, strength=0.5):
        if tag1 in self.nodes and tag2 in self.nodes:
            cross = strength * np.mean(self.nodes[tag2], axis=0)
            self.nodes[tag1].append(cross)
            self.trails[tag1].append(cross)

    def summary(self):
        return {tag: len(self.nodes[tag]) for tag in self.nodes}

# Example usage
if __name__ == "__main__":
    me = MycelialEngine()
    vec = np.random.rand(128)
    me.grow_path("intuition", vec)
    me.spore_seed("memory", vec * 0.8)
    me.cross_connect("intuition", "memory")
    echo = me.echo_query("intuition")
    print("Echo length:", np.linalg.norm(echo))
    print("Summary:", me.summary())