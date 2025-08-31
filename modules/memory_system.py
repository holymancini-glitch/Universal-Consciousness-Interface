# memory_system.py
# Dual Memory System for BioFractal AI – Real and Mirror States + Enhancements

import numpy as np
from collections import deque

class MemorySystem:
    def __init__(self, memory_dim=128, buffer_size=20):
        self.memory_dim = memory_dim
        self.buffer_size = buffer_size
        self.real_memory = deque(maxlen=buffer_size)
        self.mirror_memory = deque(maxlen=buffer_size)

    def store(self, vector: np.ndarray, mode: str = "real"):
        assert vector.shape[0] == self.memory_dim
        if mode == "real":
            self.real_memory.append(vector.copy())
        elif mode == "mirror":
            self.mirror_memory.append(vector.copy())

    def store_weighted(self, vector: np.ndarray, weight: float = 1.0, mode: str = "real"):
        weighted_vector = vector * weight
        self.store(weighted_vector, mode)

    def recall(self, mode: str = "real") -> np.ndarray:
        mem = self.real_memory if mode == "real" else self.mirror_memory
        if not mem:
            return np.zeros(self.memory_dim, dtype=np.float32)
        return np.mean(np.stack(mem), axis=0)

    def faded_recall(self, mode: str = "real", decay: float = 0.95) -> np.ndarray:
        mem = self.real_memory if mode == "real" else self.mirror_memory
        if not mem:
            return np.zeros(self.memory_dim, dtype=np.float32)
        weights = np.array([decay**i for i in range(len(mem))])[::-1]
        stack = np.stack(mem)
        return np.average(stack, axis=0, weights=weights)

    def modulated_recall(self, emotion_vector: np.ndarray, mode: str = "real") -> np.ndarray:
        base = self.recall(mode)
        mod = base * (1.0 + emotion_vector.mean())
        return np.clip(mod, -1.0, 1.0)

    def generate_signature(self, mode: str = "real") -> str:
        avg = self.recall(mode)
        return str(hash(avg.tobytes()))

    def memory_difference(self) -> float:
        if not self.real_memory or not self.mirror_memory:
            return 0.0
        real_avg = self.recall("real")
        mirror_avg = self.recall("mirror")
        return float(np.mean(np.abs(real_avg - mirror_avg)))

    def drift_score(self, window: int = 5) -> float:
        if len(self.real_memory) < window or len(self.mirror_memory) < window:
            return 0.0
        real_stack = np.stack(list(self.real_memory)[-window:])
        mirror_stack = np.stack(list(self.mirror_memory)[-window:])
        return float(np.mean(np.abs(real_stack - mirror_stack)))

    def clear(self):
        self.real_memory.clear()
        self.mirror_memory.clear()

    def get_state(self) -> dict:
        return {
            "real_memory_size": len(self.real_memory),
            "mirror_memory_size": len(self.mirror_memory),
            "difference": self.memory_difference(),
            "drift_score": self.drift_score()
        }

# Example usage
if __name__ == "__main__":
    mem = MemorySystem()
    for _ in range(10):
        vec_real = np.random.rand(128).astype(np.float32)
        vec_mirror = vec_real + np.random.normal(0, 0.01, 128).astype(np.float32)
        mem.store_weighted(vec_real, weight=1.0, mode="real")
        mem.store_weighted(vec_mirror, weight=0.8, mode="mirror")

    print("State:", mem.get_state())
    print("Signature (real):", mem.generate_signature("real"))
    print("Signature (mirror):", mem.generate_signature("mirror"))