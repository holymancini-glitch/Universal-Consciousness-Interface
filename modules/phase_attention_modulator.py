# phase_attention_modulator.py — Modulates Attention by Internal Phase State

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional

class PhaseAttentionModulator(nn.Module):
    """Phase Attention Modulator - Modulates attention by internal phase state."""
    
    def __init__(self, hidden_size: int, phase_vector_dim: int = 8):
        """
        Initialize the Phase Attention Modulator.
        
        Args:
            hidden_size: Size of the hidden state
            phase_vector_dim: Dimension of the phase vector
        """
        super(PhaseAttentionModulator, self).__init__()
        self.hidden_size = hidden_size
        self.phase_vector_dim = phase_vector_dim
        
        # Phase embedding network
        self.phase_embedding = nn.Linear(phase_vector_dim, hidden_size)
        
        # Attention modulation gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Entropy delta processing
        self.entropy_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, hidden_size),
            nn.Tanh()
        )
        
        # Phase state tracking
        self.current_phase = torch.zeros(phase_vector_dim)
        self.phase_history = []
        
    def forward(self, attention_weights: torch.Tensor, 
                hidden_state: torch.Tensor, 
                phase_vector: torch.Tensor,
                entropy_delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Phase Attention Modulator.
        
        Args:
            attention_weights: Input attention weights (batch, seq_len)
            hidden_state: Hidden state (batch, hidden_size)
            phase_vector: Phase vector (batch, phase_vector_dim)
            entropy_delta: Entropy delta for adaptive focus (optional)
            
        Returns:
            modulated_weights: Modulated attention weights
            mod_gate: Modulation gate values
        """
        # Store current phase
        self.current_phase = phase_vector.detach().cpu()
        self.phase_history.append(self.current_phase.numpy())
        
        # Encode phase vector
        phase_encoded = self.phase_embedding(phase_vector)  # (batch, hidden_size)
        
        # Combine with hidden state
        combined = torch.cat((hidden_state, phase_encoded), dim=-1)  # (batch, hidden_size * 2)
        
        # Apply modulation gate
        mod_gate = self.gate(combined)  # (batch, 1)
        
        # Modulate attention weights
        modulated_weights = attention_weights * mod_gate  # scaled attention
        
        # Apply entropy delta modulation if provided
        if entropy_delta is not None:
            entropy_mod = self.entropy_processor(entropy_delta.unsqueeze(-1))  # (batch, hidden_size)
            entropy_gate = torch.sigmoid(entropy_mod.mean(dim=-1, keepdim=True))  # (batch, 1)
            modulated_weights = modulated_weights * entropy_gate
            
        return modulated_weights, mod_gate
    
    def get_current_phase(self) -> np.ndarray:
        """Get the current phase vector."""
        return self.current_phase.numpy()
    
    def get_phase_coherence(self) -> float:
        """Calculate coherence of phase history."""
        if len(self.phase_history) < 2:
            return 1.0
        
        # Calculate coherence as average similarity between consecutive phases
        similarities = []
        for i in range(1, len(self.phase_history)):
            phase1 = self.phase_history[i-1]
            phase2 = self.phase_history[i]
            # Cosine similarity
            dot_product = np.dot(phase1, phase2)
            norms = np.linalg.norm(phase1) * np.linalg.norm(phase2)
            if norms > 0:
                similarity = dot_product / norms
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    def adapt_phase_dynamics(self, coherence: float, learning_rate: float = 0.01):
        """
        Adapt phase dynamics based on coherence feedback.
        
        Args:
            coherence: Current phase coherence
            learning_rate: Learning rate for adaptation
        """
        # Adjust phase embedding based on coherence
        if coherence < 0.5:  # Low coherence
            # Increase phase sensitivity
            for param in self.phase_embedding.parameters():
                param.data *= (1.0 + learning_rate)
        elif coherence > 0.8:  # High coherence
            # Stabilize phase dynamics
            for param in self.phase_embedding.parameters():
                param.data *= (1.0 - learning_rate * 0.1)
    
    def reset(self):
        """Reset the phase attention modulator."""
        self.current_phase = torch.zeros(self.phase_vector_dim)
        self.phase_history.clear()

class AdaptivePhaseController:
    """Controls phase dynamics for adaptive focus."""
    
    def __init__(self, phase_dim: int = 8):
        """
        Initialize the Adaptive Phase Controller.
        
        Args:
            phase_dim: Dimension of the phase vector
        """
        self.phase_dim = phase_dim
        self.phase_state = np.zeros(phase_dim)
        self.phase_velocity = np.zeros(phase_dim)
        self.phase_acceleration = np.zeros(phase_dim)
        
        # Control parameters
        self.damping = 0.9
        self.stiffness = 0.1
        self.mass = 1.0
        
        # Target phase
        self.target_phase = None
        
    def update_phase(self, entropy_delta: float, external_forces: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update phase state based on entropy delta and external forces.
        
        Args:
            entropy_delta: Entropy change for phase modulation
            external_forces: External forces acting on the phase (optional)
            
        Returns:
            updated_phase: Updated phase vector
        """
        # Calculate forces
        if self.target_phase is not None:
            # Spring force toward target
            spring_force = self.stiffness * (self.target_phase - self.phase_state)
        else:
            spring_force = np.zeros(self.phase_dim)
        
        # Entropy-driven force
        entropy_force = np.random.randn(self.phase_dim) * abs(entropy_delta)
        
        # Total force
        total_force = spring_force + entropy_force
        if external_forces is not None:
            total_force += external_forces
            
        # Update dynamics (simple mass-spring-damper system)
        self.phase_acceleration = total_force / self.mass
        self.phase_velocity = self.damping * self.phase_velocity + self.phase_acceleration
        self.phase_state = self.phase_state + self.phase_velocity
        
        # Normalize phase state
        norm = np.linalg.norm(self.phase_state)
        if norm > 0:
            self.phase_state = self.phase_state / norm
            
        return self.phase_state.copy()
    
    def set_target_phase(self, target: np.ndarray):
        """Set target phase for the controller."""
        assert len(target) == self.phase_dim
        self.target_phase = target.copy()
    
    def get_phase_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Get phase as a torch tensor."""
        return torch.FloatTensor(self.phase_state).to(device)
    
    def reset(self):
        """Reset the phase controller."""
        self.phase_state = np.zeros(self.phase_dim)
        self.phase_velocity = np.zeros(self.phase_dim)
        self.phase_acceleration = np.zeros(self.phase_dim)
        self.target_phase = None

# Example usage
if __name__ == "__main__":
    # Initialize phase attention modulator
    modulator = PhaseAttentionModulator(hidden_size=128, phase_vector_dim=8)
    
    # Create test inputs
    batch_size = 4
    seq_len = 10
    attention_weights = torch.randn(batch_size, seq_len)
    hidden_state = torch.randn(batch_size, 128)
    phase_vector = torch.randn(batch_size, 8)
    entropy_delta = torch.randn(batch_size)
    
    # Apply modulation
    modulated_weights, mod_gate = modulator(
        attention_weights, 
        hidden_state, 
        phase_vector,
        entropy_delta
    )
    
    print(f"Original attention weights shape: {attention_weights.shape}")
    print(f"Modulated attention weights shape: {modulated_weights.shape}")
    print(f"Modulation gate shape: {mod_gate.shape}")
    
    # Test phase coherence
    coherence = modulator.get_phase_coherence()
    print(f"Phase coherence: {coherence:.4f}")
    
    # Test adaptive phase controller
    controller = AdaptivePhaseController(phase_dim=8)
    entropy_delta_val = 0.5
    updated_phase = controller.update_phase(entropy_delta_val)
    print(f"Updated phase shape: {updated_phase.shape}")
    print(f"Phase norm: {np.linalg.norm(updated_phase):.4f}")