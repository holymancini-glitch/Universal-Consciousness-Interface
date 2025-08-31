# dream_feedback_types.py — Data structure for GRU-LSTM dream feedback

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class DreamFeedback:
    context: np.ndarray               # fused latent representation
    mood_index: float                # scalar mood index (0–1)
    gru_attention: np.ndarray        # GRU attention vector
    lstm_attention: np.ndarray       # LSTM attention vector
    dream_state_probs: List[float]   # [echo, archetype, intent, anomaly]
    symbolic_embedding: np.ndarray   # tag embedding (128-d)