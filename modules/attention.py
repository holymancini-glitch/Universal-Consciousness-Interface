# attention.py
# Attention Module with Mirror Awareness for BioFractal AI v2.1 + Enhancements

import numpy as np

class Attention:
    def __init__(self, shape=(64, 64, 8), mirror_enabled=True):
        self.shape = shape
        self.focus_map = np.zeros(shape, dtype=np.float32)
        self.mirror_map = np.zeros(shape, dtype=np.float32)
        self.mode = "real"  # or "mirror"
        self.mirror_enabled = mirror_enabled
        self.history = []  # attention history

    def focus(self, x: int, y: int, intensity: float = 1.0):
        if self.mode == "real":
            self.focus_map = np.zeros(self.shape)
            self.focus_map[x, y, :] = intensity
        elif self.mirror_enabled:
            self.mirror_map = np.zeros(self.shape)
            self.mirror_map[x, y, :] = intensity
        self.history.append((x, y, intensity))

    def diffuse(self, decay: float = 0.9):
        if self.mode == "real":
            self.focus_map *= decay
        elif self.mirror_enabled:
            self.mirror_map *= decay

    def switch_mode(self, mode: str):
        assert mode in ["real", "mirror"]
        self.mode = mode

    def get_attention(self) -> np.ndarray:
        return self.focus_map if self.mode == "real" else self.mirror_map

    def harmonize_attention(self, alpha: float = 0.5):
        if not self.mirror_enabled:
            return
        blended = alpha * self.focus_map + (1 - alpha) * self.mirror_map
        self.focus_map = blended
        self.mirror_map = blended

    def apply_emotional_modulation(self, emotion_vector: np.ndarray):
        scalar = 1.0 + emotion_vector.mean()
        if self.mode == "real":
            self.focus_map *= scalar
        else:
            self.mirror_map *= scalar

    def sync_with_latent(self, latent_map: np.ndarray):
        average_map = latent_map.mean(axis=2, keepdims=True)
        if self.mode == "real":
            self.focus_map = np.repeat(average_map, self.shape[2], axis=2)
        else:
            self.mirror_map = np.repeat(average_map, self.shape[2], axis=2)

    def modulate_by_mirror_error(self, error_score: float):
        if not self.mirror_enabled:
            return
        self.mirror_map *= 1.0 + error_score

    def focus_by_entropy(self, data_grid: np.ndarray):
        entropy = -data_grid * np.log(data_grid + 1e-6)
        entropy_sum = entropy.sum(axis=2)
        max_entropy_loc = np.unravel_index(np.argmax(entropy_sum), self.shape[:2])
        self.focus(*max_entropy_loc, intensity=1.0)

    def reset(self):
        self.focus_map = np.zeros(self.shape, dtype=np.float32)
        self.mirror_map = np.zeros(self.shape, dtype=np.float32)
        self.mode = "real"
        self.history = []

# Example usage
if __name__ == "__main__":
    attn = Attention()
    dummy_latent = np.random.rand(64, 64, 8).astype(np.float32)
    attn.sync_with_latent(dummy_latent)
    attn.apply_emotional_modulation(np.random.rand(10))
    attn.focus_by_entropy(dummy_latent)
    attn.harmonize_attention()
    attn.modulate_by_mirror_error(0.12)
    print("Attention map sum:", np.sum(attn.get_attention()))