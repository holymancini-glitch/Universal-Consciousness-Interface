# fractall_ai.py
# Central Decision Engine for BioFractal AI – Dual-Trajectory Logic with Enhancements

import numpy as np

class FractallAI:
    def __init__(self, input_dim=128, output_dim=64):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights_real = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        self.weights_mirror = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.1
        self.mode = "real"  # or "mirror"
        self.last_real_output = None
        self.last_mirror_output = None
        self.last_input = None

    def process(self, input_vector: np.ndarray) -> np.ndarray:
        assert input_vector.shape[0] == self.input_dim
        self.last_input = input_vector
        if self.mode == "real":
            output = np.tanh(input_vector @ self.weights_real)
            self.last_real_output = output
        else:
            output = np.tanh(input_vector @ self.weights_mirror)
            self.last_mirror_output = output
        return output

    def dual_trajectory(self, input_vector: np.ndarray) -> dict:
        assert input_vector.shape[0] == self.input_dim
        real_out = np.tanh(input_vector @ self.weights_real)
        mirror_out = np.tanh(input_vector @ self.weights_mirror)
        self.last_real_output = real_out
        self.last_mirror_output = mirror_out
        self.last_input = input_vector
        divergence = np.mean(np.abs(real_out - mirror_out))
        return {
            "real": real_out,
            "mirror": mirror_out,
            "divergence": divergence
        }

    def condition_with_emotion(self, emotion_vector: np.ndarray):
        mod = 1.0 + emotion_vector.mean()
        self.weights_real *= mod
        self.weights_mirror *= (2.0 - mod)

    def adapt(self, input_vector: np.ndarray, target_vector: np.ndarray, lr=0.01):
        pred = self.process(input_vector)
        error = target_vector - pred
        if self.mode == "real":
            self.weights_real += lr * np.outer(input_vector, error)
        else:
            self.weights_mirror += lr * np.outer(input_vector, error)

    def auto_switch_by_divergence(self, threshold=0.25, input_vector=None):
        if input_vector is not None:
            divergence = self.dual_trajectory(input_vector)["divergence"]
            if divergence > threshold:
                self.switch_mode("mirror" if self.mode == "real" else "real")

    def qubit_mode_output(self, alpha=0.5):
        if self.last_real_output is not None and self.last_mirror_output is not None:
            return alpha * self.last_real_output + (1 - alpha) * self.last_mirror_output
        else:
            return np.zeros(self.output_dim, dtype=np.float32)

    def switch_mode(self, mode: str):
        assert mode in ["real", "mirror"]
        self.mode = mode

    def reset(self):
        self.weights_real = np.random.randn(self.input_dim, self.output_dim).astype(np.float32) * 0.1
        self.weights_mirror = np.random.randn(self.input_dim, self.output_dim).astype(np.float32) * 0.1
        self.mode = "real"
        self.last_real_output = None
        self.last_mirror_output = None
        self.last_input = None

# Example usage
if __name__ == "__main__":
    ai = FractallAI()
    vec = np.random.rand(128).astype(np.float32)
    output = ai.dual_trajectory(vec)
    ai.condition_with_emotion(np.random.rand(10).astype(np.float32))
    ai.adapt(vec, np.random.rand(64).astype(np.float32))
    ai.auto_switch_by_divergence(threshold=0.2, input_vector=vec)
    blended = ai.qubit_mode_output(alpha=0.6)
    print("Divergence:", output["divergence"])
    print("Qubit blended output mean:", np.mean(blended))