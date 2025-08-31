
import torch
import time

from core.mycelial import MycelialEngine
from latent_space import LatentSpace
from attention import AttentionField
from fractal_ai import FractalAI
from feedback import FeedbackLoop
from self_model import SelfModel
from cortex import LatentToPromptMLP, LanguageCortex
from cohesion_layer import CohesionLayer
from semantic_memory import SemanticMemory
from narrative_loop import NarrativeLoop
from temporal_memory import TemporalMemoryGRU

def run_orchestration(steps=10, latent_dim=256):
    latent = LatentSpace(latent_dim)
    attention = AttentionField(latent)
    ai = FractalAI(latent, attention)
    feedback = FeedbackLoop()
    self_model = SelfModel()
    encoder = LatentToPromptMLP(latent_dim=latent_dim)
    cortex = LanguageCortex(encoder)

    cohesion = CohesionLayer()
    memory = SemanticMemory()
    narrative = NarrativeLoop(self_model, memory)
    temporal_memory = TemporalMemoryGRU(input_dim=latent_dim, hidden_dim=128)
    temporal_memory.reset_state()

    for step in range(steps):
        print(f"\n[Step {step+1}] —–––––––––––––––––––––––––––––––––––––––––––––––––")
        # 1. Стимул
        spike = torch.randn(latent_dim)
        latent.store(spike)

        # 2. Фокус
        focus = attention.focus(spike)

        # 3. Прогноз
        prediction = ai.predict(spike)

        # 4. Помилка + адаптація
        error = feedback.compute_error(prediction, focus)
        adapted = feedback.adapt(focus, error)

        # 5. Self-модель
        entropy = latent.get_entropy()
        self_model.update(adapted, entropy)
        self_state = self_model.describe()

        # 6. Мовна реакція
        phrase = cortex.interpret(adapted)

        # 7. Зберігання у Semantic Memory
        memory.store(adapted, entropy, phrase)

        # 8. Обчислення Cohesion
        cohesion_val = cohesion.observe(focus.norm(), entropy, error)
        cohesion_avg = cohesion.mean_cohesion()

        # 9. GRU-памʼять
        adapted_seq = adapted.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, latent_dim)
        gru_out, hidden = temporal_memory(adapted_seq)
        hidden_norm = torch.norm(hidden[0])  # Візуалізуємо hidden[0]

        # 10. Наратив
        story = narrative.narrate()

        print(f"Focus Norm: {focus.norm():.4f} | Error: {error:.4f}")
        print(f"Entropy: {entropy:.2f} | Cohesion: {cohesion_val:.2f} (avg: {cohesion_avg:.2f})")
        print(f"Hidden State Norm (GRU): {hidden_norm:.4f}")
        print(f"LLM Prompt: {phrase}")
        print(f"Story: {story}")

        time.sleep(0.5)

if __name__ == '__main__':
    run_orchestration(steps=10)
