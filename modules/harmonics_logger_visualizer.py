# harmonics_logger_visualizer.py — BioFractal AI v3.3

import matplotlib.pyplot as plt
import numpy as np

class HarmonicsLogger:
    def __init__(self):
        self.tick_history = []
        self.delta_history = []
        self.mod_vector_history = []

    def log(self, event_tick, delta, mod_vector):
        self.tick_history.append(event_tick)
        self.delta_history.append(delta.item() if hasattr(delta, 'item') else delta)
        mod_norm = float(mod_vector.norm().item()) if hasattr(mod_vector, 'norm') else np.linalg.norm(mod_vector)
        self.mod_vector_history.append(mod_norm)

    def visualize(self, show=True, save_path=None):
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.tick_history, self.delta_history, label='Entropy Δ', color='purple')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
        plt.title('Entropy Delta over Time')
        plt.ylabel('Δ Entropy')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.tick_history, self.mod_vector_history, label='Modulation Vector Norm', color='blue')
        plt.title('Harmonic Modulation Intensity')
        plt.xlabel('Event Tick')
        plt.ylabel('||Mod Vector||')
        plt.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

# Example usage:
# logger = HarmonicsLogger()
# logger.log(current_tick, entropy_delta, mod_vector)
# logger.visualize()