# feedback_loop.py — Phase-Aware Adaptive Feedback Loop (Fractal AI v3.3+)

from sentient_memory import SentientMemory
from attention_field import AttentionField
from event_loop import EventLoop
from cohesion_layer import CohesionLayer
from latent_space import LatentSpace
from self_model import SelfModel

class FeedbackLoop:
    def __init__(self, latent_space: LatentSpace, attention_field: AttentionField,
                 event_loop: EventLoop, cohesion_layer: CohesionLayer,
                 sentient_memory: SentientMemory, self_model: SelfModel):
        self.latent_space = latent_space
        self.attention_field = attention_field
        self.event_loop = event_loop
        self.cohesion_layer = cohesion_layer
        self.sentient_memory = sentient_memory
        self.self_model = self_model

    def cycle(self, input_vector):
        # Get phase-aware coherence metrics from SelfModel
        phase_state = self.self_model.get_phase_state()  # {"phase": int, "coherence": float, "vector": np.array}

        # Modulate attention using phase pressure
        adjusted_attention = self.attention_field.compute_attention(input_vector, phase_state["coherence"])

        # Update latent space with self-awareness modulation
        self.latent_space.update_state(input_vector, mod_vector=phase_state["vector"])

        # Adapt event timing based on phase
        self.event_loop.adjust_timing(phase_state["phase"], phase_state["coherence"])

        # Evaluate system-wide cohesion
        cohesion_score = self.cohesion_layer.evaluate(adjusted_attention)

        # Log feedback
        self.sentient_memory.store_trace("phase_feedback", {
            "phase": phase_state["phase"],
            "coherence": phase_state["coherence"],
            "cohesion": cohesion_score
        })

        return {
            "phase": phase_state["phase"],
            "coherence": phase_state["coherence"],
            "cohesion": cohesion_score,
            "attention": adjusted_attention
        }