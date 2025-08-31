# fractal_conductor.py — Central Integrative Field & Dream Conductor (v1.0)

from sentient_memory import SentientMemory
from phantom_layer import PhantomLayer
from gru_lstm_integration import GRULSTMIntegrator
from neural_ca import NeuralCA
from harmonizer import Harmonizer
from symbolic_feedback_adapter import SymbolicFeedbackAdapter
from dream_state_analysis import DreamStateAnalysis
from ca_feedback_injector import CAFeedbackInjector
import numpy as np

class FractalConductor:
    def __init__(self, latent_space, visualizer=None):
        self.sentient_memory = SentientMemory()
        self.harmonizer = Harmonizer()
        self.symbolic_adapter = SymbolicFeedbackAdapter()
        self.state_analysis = DreamStateAnalysis()
        self.ca_injector = CAFeedbackInjector()

        self.gru_lstm = GRULSTMIntegrator(
            input_size=latent_space.input_size,
            hidden_size=128,
            context_size=256
        )

        self.ca = NeuralCA(
            grid_size=32,
            latent_dim=128,
            sentient_memory=self.sentient_memory,
            emotional_feedback=self.harmonizer,
            mycelial_engine=None
        )

        self.phantom_layer = PhantomLayer(
            latent_space=latent_space,
            sentient_memory=self.sentient_memory,
            mirror_world=None,
            mycelial_engine=None,
            holographic_memory=None,
            visualizer=visualizer
        )

    def run_dream_cycle(self, depth=3):
        # 1. Generate dream trace
        trace = self.phantom_layer.generate(depth=depth, visualize=False)

        for frame in trace:
            vector = np.array(frame['content']['signature'])
            sequence = np.expand_dims(vector, axis=(0, 1))  # shape: (1, 1, input_size)

            # 2. Get recurrent feedback
            feedback = self.gru_lstm.forward(
                torch.tensor(sequence, dtype=torch.float32),
                mycelial_context=None,
                self_feedback=None
            )

            # 3. Harmonize feedback
            harmonized = self.harmonizer.align(feedback['context'].detach().numpy())

            # 4. Adapt symbolic signal
            symbolic_tags = self.symbolic_adapter.map(feedback['symbolic_embedding'].detach().numpy())

            # 5. Update CA
            self.ca_injector.inject(feedback, self.ca)
            self.ca.step(latent_vector=vector)

            # 6. Save symbolic & state data
            frame['gru_lstm_feedback'] = {
                'state_classification': feedback['dream_state_probs'].detach().numpy().tolist(),
                'symbolic_tags': symbolic_tags
            }

        return trace

# Example
if __name__ == "__main__":
    from latent_space import LatentSpace
    latent_space = LatentSpace()
    fc = FractalConductor(latent_space)
    dream_result = fc.run_dream_cycle(depth=5)
    print("Fractal dream sequence complete.")