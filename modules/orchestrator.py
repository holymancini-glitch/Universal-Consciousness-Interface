# orchestrator.py — BioFractal AI v3.3

from modules.latent_space import LatentSpace
from modules.entropy_harmonizer import EntropyHarmonizer
from modules.cohesion_layer import CohesionLayer
from modules.event_loop import EventLoop
from modules.feedback_loop import FeedbackLoop
from modules.self_model import SelfModel
from modules.harmonics_logger import HarmonicsLogger

class BioFractalOrchestrator:
    def __init__(self, latent_dim=128):
        self.latent_space = LatentSpace(dim=latent_dim)
        self.entropy_harmonizer = EntropyHarmonizer(latent_dim=latent_dim)
        self.cohesion_layer = CohesionLayer()
        self.feedback_loop = FeedbackLoop()
        self.self_model = SelfModel(latent_dim=latent_dim)
        self.logger = HarmonicsLogger()
        self.event_loop = EventLoop(callback=self.step)

        self.tick = 0
        self.entropy_score = 0.0

    def step(self):
        # Update harmonic modulation
        modulation, delta = self.entropy_harmonizer(self.entropy_score, self.tick)
        self.latent_space.inject_modulation(modulation)

        # Update internal coherence
        self.cohesion_layer.adjust_from_entropy(delta)
        coherence = self.cohesion_layer.get_coherence()

        # Feedback & self-model update
        self.feedback_loop.update(self.latent_space, coherence)
        self.self_model.update(self.latent_space, coherence)

        # Logging
        self.logger.log(self.tick, self.entropy_score, delta.item(), coherence)

        # Advance tick
        self.tick += 1
        self.entropy_score = self.latent_space.estimate_entropy()

    def run(self, steps=100):
        self.event_loop.run(steps)

# Example
if __name__ == "__main__":
    orchestrator = BioFractalOrchestrator()
    orchestrator.run(steps=200)