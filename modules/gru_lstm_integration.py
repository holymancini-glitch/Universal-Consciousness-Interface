# gru_lstm_integration.py — BioFractal AI v3.3 Phase-Aware Enhanced Recurrent Module

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs):
        attn_weights = torch.softmax(self.attn(encoder_outputs).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights

class GRULSTMIntegrator(nn.Module):
    def __init__(self, input_size, hidden_size, context_size):
        super(GRULSTMIntegrator, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)

        self.fusion_layer = nn.Linear(hidden_size * 2, context_size)
        self.dropout = nn.Dropout(0.2)

        self.phase_embedding = nn.Embedding(6, context_size)
        self.phase_modulator = nn.Linear(context_size, context_size)

        self.mood_proj = nn.Linear(hidden_size, 1)
        self.mycelial_gate = nn.Linear(context_size, 1)
        self.dream_classifier = nn.Linear(context_size, 4)
        self.temporal_shift = nn.Parameter(torch.randn(context_size))
        self.self_alignment_gate = nn.Linear(context_size, context_size)
        self.tag_activator = nn.Linear(hidden_size, 128)

    def forward(self, sequence, phase_index, mycelial_context=None, reset_states=False, self_feedback=None):
        if reset_states:
            h0 = torch.zeros(1, sequence.size(0), self.hidden_size, device=sequence.device)
            c0 = torch.zeros(1, sequence.size(0), self.hidden_size, device=sequence.device)
        else:
            h0, c0 = None, None

        gru_out, _ = self.gru(sequence, h0)
        lstm_out, _ = self.lstm(sequence, (h0, c0) if h0 is not None else None)

        attn_gru, _ = self.attention(gru_out)
        attn_lstm, _ = self.attention(lstm_out)

        fusion = torch.cat((attn_gru, attn_lstm), dim=-1)
        fused = torch.tanh(self.fusion_layer(fusion))
        spiral_context = fused + torch.sin(self.temporal_shift)

        phase_emb = self.phase_embedding(phase_index)
        phase_tuned = spiral_context + self.phase_modulator(phase_emb)

        if mycelial_context is not None:
            gate = torch.sigmoid(self.mycelial_gate(mycelial_context))
            phase_tuned = phase_tuned * gate

        if self_feedback is not None:
            delta = torch.tanh(self.self_alignment_gate(self_feedback - phase_tuned))
            phase_tuned = phase_tuned + delta

        mood = torch.sigmoid(self.mood_proj(attn_gru))
        dream_logits = self.dream_classifier(phase_tuned)
        dream_state = torch.softmax(dream_logits, dim=-1)
        tag_embedding = self.tag_activator(attn_gru)

        return {
            'context': phase_tuned,
            'mood_index': mood.squeeze(-1),
            'gru_attention': attn_gru,
            'lstm_attention': attn_lstm,
            'dream_state_probs': dream_state,
            'symbolic_embedding': tag_embedding
        }