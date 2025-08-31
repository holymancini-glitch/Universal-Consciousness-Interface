# fep_neural_model.py
# FEPNeuralModel for CL1 biological processor simulation

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class NeuralState:
    """Represents the state of the biological neural network."""
    membrane_potential: np.ndarray
    synaptic_weights: np.ndarray
    firing_rates: np.ndarray
    prediction_error: float
    free_energy: float

class FEPNeuralModel:
    """FEPNeuralModel - Biological computing substrate using the Free Energy Principle."""
    
    def __init__(self, num_neurons: int = 800000, input_dim: int = 1024, output_dim: int = 512):
        """
        Initialize the FEP Neural Model.
        
        Args:
            num_neurons: Number of neurons in the model (~800,000 for CL1)
            input_dim: Dimension of input space
            output_dim: Dimension of output space
        """
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Neural state variables
        self.membrane_potential = np.random.randn(num_neurons) * 0.1
        self.synaptic_weights = np.random.randn(num_neurons, num_neurons) * 0.01
        self.firing_rates = np.zeros(num_neurons)
        self.prediction_error = 0.0
        self.free_energy = 0.0
        
        # Internal models for prediction
        self.internal_model = self._create_internal_model()
        self.sensory_precision = 1.0  # Precision weighting for sensory inputs
        
        # History tracking
        self.state_history = deque(maxlen=1000)
        self.free_energy_history = deque(maxlen=10000)
        
        # Learning parameters
        self.learning_rate = 0.001
        self.attention_gain = 1.0
        
    def _create_internal_model(self) -> nn.Module:
        """Create internal predictive model."""
        model = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, self.input_dim)
        )
        return model
    
    def compute_firing_rates(self, stimulus: np.ndarray) -> np.ndarray:
        """
        Compute neural firing rates based on membrane potential and stimulus.
        
        Args:
            stimulus: Input stimulus vector
            
        Returns:
            firing_rates: Computed firing rates for each neuron
        """
        # Simple leaky integrate-and-fire model
        decay = 0.9
        self.membrane_potential = decay * self.membrane_potential + stimulus[:self.num_neurons] if len(stimulus) >= self.num_neurons else np.pad(stimulus, (0, self.num_neurons - len(stimulus)))
        
        # Apply activation function (spiking model)
        threshold = 0.5
        self.firing_rates = np.where(self.membrane_potential > threshold, 
                                   self.membrane_potential - threshold, 0)
        
        return self.firing_rates
    
    def predict(self, input_state: np.ndarray) -> np.ndarray:
        """
        Generate prediction using internal model.
        
        Args:
            input_state: Current input state
            
        Returns:
            prediction: Predicted next state
        """
        # Convert to torch tensor
        input_tensor = torch.FloatTensor(input_state).unsqueeze(0)
        
        # Generate prediction
        with torch.no_grad():
            prediction = self.internal_model(input_tensor)
            
        return prediction.squeeze().numpy()
    
    def compute_prediction_error(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Compute prediction error between actual and predicted states.
        
        Args:
            actual: Actual observed state
            predicted: Predicted state
            
        Returns:
            prediction_error: Scalar prediction error
        """
        error = np.mean((actual - predicted) ** 2)
        return error
    
    def compute_free_energy(self, prediction_error: float, prior_precision: float = 1.0) -> float:
        """
        Compute free energy as a bound on surprise.
        
        Args:
            prediction_error: Current prediction error
            prior_precision: Precision of prior expectations
            
        Returns:
            free_energy: Computed free energy
        """
        # Free energy is a combination of prediction error and prior expectations
        free_energy = 0.5 * (prior_precision * prediction_error + 
                           np.sum((self.synaptic_weights ** 2)))
        return free_energy
    
    def update_synaptic_weights(self, input_state: np.ndarray, learning_rate: Optional[float] = None):
        """
        Update synaptic weights using Hebbian learning modulated by prediction error.
        
        Args:
            input_state: Current input state
            learning_rate: Optional learning rate (uses default if None)
        """
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Hebbian learning with prediction error modulation
        activity_outer = np.outer(self.firing_rates, self.firing_rates)
        error_modulation = np.exp(-self.prediction_error)  # Less learning when errors are high
        
        # Update weights
        self.synaptic_weights += lr * error_modulation * activity_outer
        
        # Apply constraints to keep weights bounded
        self.synaptic_weights = np.clip(self.synaptic_weights, -1.0, 1.0)
    
    def minimize_free_energy(self, input_state: np.ndarray, steps: int = 10):
        """
        Minimize free energy through active inference.
        
        Args:
            input_state: Current input state
            steps: Number of minimization steps
        """
        for _ in range(steps):
            # Generate prediction
            prediction = self.predict(input_state)
            
            # Compute prediction error
            self.prediction_error = self.compute_prediction_error(input_state, prediction)
            
            # Compute free energy
            self.free_energy = self.compute_free_energy(self.prediction_error)
            
            # Update synaptic weights
            self.update_synaptic_weights(input_state)
            
            # Store in history
            self.free_energy_history.append(self.free_energy)
    
    def process_stimulus(self, stimulus: np.ndarray, minimize_fe: bool = True) -> Dict[str, np.ndarray]:
        """
        Process incoming stimulus through the FEP neural model.
        
        Args:
            stimulus: Input stimulus vector
            minimize_fe: Whether to perform free energy minimization
            
        Returns:
            output: Dictionary containing processing results
        """
        # Ensure stimulus is the right size
        if len(stimulus) < self.input_dim:
            stimulus = np.pad(stimulus, (0, self.input_dim - len(stimulus)))
        elif len(stimulus) > self.input_dim:
            stimulus = stimulus[:self.input_dim]
        
        # Compute firing rates
        firing_rates = self.compute_firing_rates(stimulus)
        
        # Generate prediction
        prediction = self.predict(stimulus)
        
        # Compute prediction error
        prediction_error = self.compute_prediction_error(stimulus, prediction)
        self.prediction_error = prediction_error
        
        # Compute free energy
        free_energy = self.compute_free_energy(prediction_error)
        self.free_energy = free_energy
        
        # Minimize free energy if requested
        if minimize_fe:
            self.minimize_free_energy(stimulus)
        
        # Generate output (simplified)
        # In a real implementation, this would be more complex
        output_indices = np.random.choice(self.num_neurons, self.output_dim, replace=False)
        output = self.firing_rates[output_indices]
        
        # Store current state
        current_state = NeuralState(
            membrane_potential=self.membrane_potential.copy(),
            synaptic_weights=self.synaptic_weights.copy(),
            firing_rates=self.firing_rates.copy(),
            prediction_error=self.prediction_error,
            free_energy=self.free_energy
        )
        self.state_history.append(current_state)
        
        return {
            "firing_rates": firing_rates,
            "prediction": prediction,
            "prediction_error": prediction_error,
            "free_energy": free_energy,
            "output": output,
            "state": current_state
        }
    
    def adapt_attention(self, surprise_level: float):
        """
        Adapt attention gain based on surprise level.
        
        Args:
            surprise_level: Level of surprise (higher = more surprising)
        """
        # Increase attention gain when surprised, decrease when predictable
        self.attention_gain = 1.0 + 0.5 * np.tanh(surprise_level - 1.0)
        
    def get_current_state(self) -> NeuralState:
        """Get the current neural state."""
        return NeuralState(
            membrane_potential=self.membrane_potential.copy(),
            synaptic_weights=self.synaptic_weights.copy(),
            firing_rates=self.firing_rates.copy(),
            prediction_error=self.prediction_error,
            free_energy=self.free_energy
        )
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics for monitoring."""
        return {
            "prediction_error": self.prediction_error,
            "free_energy": self.free_energy,
            "avg_free_energy": np.mean(self.free_energy_history) if self.free_energy_history else 0.0,
            "attention_gain": self.attention_gain,
            "synaptic_activity": np.mean(np.abs(self.synaptic_weights)),
            "firing_rate_mean": np.mean(self.firing_rates),
            "firing_rate_std": np.std(self.firing_rates)
        }
    
    def reset(self):
        """Reset the neural model to initial state."""
        self.membrane_potential = np.random.randn(self.num_neurons) * 0.1
        self.synaptic_weights = np.random.randn(self.num_neurons, self.num_neurons) * 0.01
        self.firing_rates = np.zeros(self.num_neurons)
        self.prediction_error = 0.0
        self.free_energy = 0.0
        self.state_history.clear()
        self.free_energy_history.clear()
        self.attention_gain = 1.0

# Example usage
if __name__ == "__main__":
    # Initialize FEP Neural Model
    fep_model = FEPNeuralModel(num_neurons=1000, input_dim=256, output_dim=128)  # Smaller for testing
    
    # Create test stimulus
    stimulus = np.random.randn(256).astype(np.float32)
    
    # Process stimulus
    result = fep_model.process_stimulus(stimulus)
    
    print(f"Prediction error: {result['prediction_error']:.4f}")
    print(f"Free energy: {result['free_energy']:.4f}")
    print(f"Output shape: {result['output'].shape}")
    print(f"Firing rates mean: {np.mean(result['firing_rates']):.4f}")
    
    # Test metrics
    metrics = fep_model.get_metrics()
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test adaptation
    fep_model.adapt_attention(2.0)
    print(f"\nAttention gain after surprise: {fep_model.attention_gain:.4f}")