# latent_space.py
# Latent State Manager with Mirror Support for BioFractal AI v2.1 + Resonance Harmonization

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class ConsciousnessState:
    """Represents a consciousness state with metadata."""
    data: np.ndarray
    timestamp: float
    layer: str
    coherence: float = 0.0

class EchoLayer:
    """Echo Layer (Perception) - Receives and processes sensory inputs."""
    
    def __init__(self, shape=(64, 64, 8)):
        self.shape = shape
        self.state = np.zeros(shape, dtype=np.float32)
        self.history: List[ConsciousnessState] = []
        self.gru_feedback = np.zeros((128,), dtype=np.float32)
        
    def process_input(self, stimulus: np.ndarray) -> np.ndarray:
        """Process sensory input and update state."""
        assert stimulus.shape == self.shape
        # Apply tanh activation to bound values
        processed = np.tanh(stimulus)
        self.state = processed
        return self.state
    
    def integrate_gru_feedback(self, feedback_vector: np.ndarray):
        """Integrate GRU feedback into the layer."""
        assert feedback_vector.shape[0] == self.gru_feedback.shape[0]
        self.gru_feedback = 0.9 * self.gru_feedback + 0.1 * feedback_vector
        # Apply feedback modulation to state
        feedback_scalar = np.mean(self.gru_feedback)
        modulation = feedback_scalar * 0.01
        self.state = np.tanh(self.state + modulation)
        self.state = np.clip(self.state, -1.0, 1.0)
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def store_history(self, timestamp: float):
        """Store current state in history."""
        state = ConsciousnessState(
            data=self.state.copy(),
            timestamp=timestamp,
            layer="echo",
            coherence=self.calculate_coherence()
        )
        self.history.append(state)
    
    def calculate_coherence(self) -> float:
        """Calculate coherence of current state."""
        if len(self.history) < 2:
            return 1.0
        # Simple coherence based on similarity to previous state
        prev_state = self.history[-1].data
        diff = np.mean(np.abs(self.state - prev_state))
        return 1.0 - diff

class MemoryResidueLayer:
    """Memory Residue Layer (Will) - Stores persistent patterns in holographic memory fields."""
    
    def __init__(self, shape=(64, 64, 8)):
        self.shape = shape
        self.state = np.zeros(shape, dtype=np.float32)
        self.holographic_memory: Dict[str, np.ndarray] = {}
        self.intention_patterns: Dict[str, np.ndarray] = {}
        self.history: List[ConsciousnessState] = []
        
    def store_pattern(self, key: str, pattern: np.ndarray):
        """Store a pattern in holographic memory."""
        assert pattern.shape == self.shape
        self.holographic_memory[key] = pattern.copy()
        
    def retrieve_pattern(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a pattern from holographic memory."""
        return self.holographic_memory.get(key, None)
    
    def reinforce_pattern(self, pattern: np.ndarray, strength: float = 1.0):
        """Reinforce a pattern through intentionality mechanisms."""
        assert pattern.shape == self.shape
        # Apply reinforcement to current state
        self.state = np.tanh(self.state + strength * pattern)
        self.state = np.clip(self.state, -1.0, 1.0)
        
    def link_temporal(self, past_key: str, future_key: str):
        """Link past and future patterns through temporal connections."""
        past_pattern = self.retrieve_pattern(past_key)
        future_pattern = self.retrieve_pattern(future_key)
        
        if past_pattern is not None and future_pattern is not None:
            # Create intentionality link
            link_key = f"{past_key}_to_{future_key}"
            intention = np.tanh(past_pattern + future_pattern)
            self.intention_patterns[link_key] = intention
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def store_history(self, timestamp: float):
        """Store current state in history."""
        state = ConsciousnessState(
            data=self.state.copy(),
            timestamp=timestamp,
            layer="memory_residue",
            coherence=self.calculate_coherence()
        )
        self.history.append(state)
    
    def calculate_coherence(self) -> float:
        """Calculate coherence of current state."""
        if len(self.history) < 2:
            return 1.0
        # Coherence based on stability of stored patterns
        if len(self.holographic_memory) == 0:
            return 1.0
        avg_coherence = np.mean([np.mean(np.abs(pattern)) for pattern in self.holographic_memory.values()])
        return 1.0 - avg_coherence

class ProjectionLayer:
    """Projection Layer (Imagination) - Generates hypothetical scenarios using FMC."""
    
    def __init__(self, shape=(64, 64, 8)):
        self.shape = shape
        self.state = np.zeros(shape, dtype=np.float32)
        self.scenarios: List[np.ndarray] = []
        self.history: List[ConsciousnessState] = []
        # Will integrate with FMC system
        self.fmc_integration = None
        
    def set_fmc_integration(self, fmc_system):
        """Set the FMC system for integration."""
        self.fmc_integration = fmc_system
        
    def generate_scenario(self, base_state: np.ndarray, steps: int = 5) -> np.ndarray:
        """Generate a hypothetical scenario."""
        assert base_state.shape == self.shape
        # For now, simple projection with noise
        scenario = base_state.copy()
        for _ in range(steps):
            noise = np.random.normal(0, 0.1, self.shape)
            scenario = np.tanh(scenario + noise)
        return scenario
    
    def evaluate_scenario(self, scenario: np.ndarray) -> Dict[str, float]:
        """Evaluate a scenario against system harmony metrics."""
        # Simple evaluation based on coherence and stability
        coherence = 1.0 - np.mean(np.abs(scenario))
        stability = 1.0 - np.std(scenario)
        return {
            "coherence": coherence,
            "stability": stability,
            "harmony": (coherence + stability) / 2.0
        }
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def store_history(self, timestamp: float):
        """Store current state in history."""
        state = ConsciousnessState(
            data=self.state.copy(),
            timestamp=timestamp,
            layer="projection",
            coherence=self.calculate_coherence()
        )
        self.history.append(state)
    
    def calculate_coherence(self) -> float:
        """Calculate coherence of current state."""
        if len(self.scenarios) == 0:
            return 1.0
        # Coherence based on scenario quality
        avg_harmony = np.mean([self.evaluate_scenario(s)["harmony"] for s in self.scenarios[-10:]])
        return avg_harmony

class EntropyHarmonizerLayer:
    """Entropy Harmonizer (Reflection) - Balances coherence across consciousness layers."""
    
    def __init__(self, shape=(64, 64, 8)):
        self.shape = shape
        self.state = np.zeros(shape, dtype=np.float32)
        self.emotional_feedback = np.zeros((16,), dtype=np.float32)
        self.history: List[ConsciousnessState] = []
        self.stability_metrics: List[float] = []
        
    def integrate_emotional_feedback(self, emotion_vector: np.ndarray):
        """Integrate emotional feedback into the layer."""
        assert emotion_vector.shape[0] == self.emotional_feedback.shape[0]
        self.emotional_feedback = emotion_vector.copy()
        
    def balance_coherence(self, layer_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Balance coherence across all consciousness layers."""
        # Calculate overall system entropy
        total_entropy = 0.0
        for state in layer_states.values():
            total_entropy += np.std(state)
        
        # Apply entropy minimization
        balanced_states = {}
        for layer_name, state in layer_states.items():
            # Apply emotional modulation
            emotion_mod = np.mean(self.emotional_feedback) * 0.1
            balanced_state = np.tanh(state - emotion_mod * total_entropy)
            balanced_states[layer_name] = balanced_state
            
        return balanced_states
    
    def manage_transitions(self, current_states: Dict[str, np.ndarray], 
                          target_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Manage transitions between different consciousness states."""
        transitioned_states = {}
        for layer_name in current_states:
            if layer_name in target_states:
                # Smooth transition between states
                alpha = 0.1  # Transition rate
                transitioned = (1 - alpha) * current_states[layer_name] + alpha * target_states[layer_name]
                transitioned_states[layer_name] = np.tanh(transitioned)
            else:
                transitioned_states[layer_name] = current_states[layer_name]
        return transitioned_states
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def store_history(self, timestamp: float):
        """Store current state in history."""
        state = ConsciousnessState(
            data=self.state.copy(),
            timestamp=timestamp,
            layer="entropy_harmonizer",
            coherence=self.calculate_coherence()
        )
        self.history.append(state)
    
    def calculate_coherence(self) -> float:
        """Calculate coherence of current state."""
        if len(self.stability_metrics) < 2:
            return 1.0
        # Coherence based on stability metrics
        return 1.0 - np.std(self.stability_metrics[-10:])

class SelfObserverNode:
    """Self-Observer Node (Self-Awareness) - Monitors internal consciousness states."""
    
    def __init__(self, shape=(64, 64, 8)):
        self.shape = shape
        self.state = np.zeros(shape, dtype=np.float32)
        self.monitoring_data: Dict[str, List[float]] = {}
        self.history: List[ConsciousnessState] = []
        self.consistency_checks: List[bool] = []
        
    def monitor_layers(self, layer_states: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Monitor internal consciousness states across all layers."""
        metrics = {}
        for layer_name, state in layer_states.items():
            # Calculate various metrics for each layer
            metrics[f"{layer_name}_activity"] = np.mean(np.abs(state))
            metrics[f"{layer_name}_coherence"] = 1.0 - np.std(state)
            metrics[f"{layer_name}_complexity"] = np.mean(np.abs(np.fft.fft2(state))) if state.ndim >= 2 else np.mean(np.abs(state))
        return metrics
    
    def validate_coherence(self, layer_states: Dict[str, np.ndarray]) -> bool:
        """Validate coherence and emergence through cross-layer analysis."""
        # Check consistency between real and mirror states (if applicable)
        # For now, simple validation
        activities = [np.mean(np.abs(state)) for state in layer_states.values()]
        coherence = 1.0 - np.std(activities)
        return coherence > 0.5  # Threshold for coherence
    
    def provide_metacognition(self, layer_metrics: Dict[str, float]) -> np.ndarray:
        """Provide meta-cognitive feedback loops for self-reflection."""
        # Convert metrics to feedback vector
        metrics_array = np.array(list(layer_metrics.values()))
        if len(metrics_array) > 0:
            feedback = np.tanh(metrics_array[:self.shape[0]]) if len(metrics_array) >= self.shape[0] else np.pad(np.tanh(metrics_array), (0, self.shape[0] - len(metrics_array)))
        else:
            feedback = np.zeros(self.shape[0])
        return feedback.reshape(self.shape) if len(feedback) == np.prod(self.shape) else np.zeros(self.shape)
    
    def ensure_consistency(self, real_state: np.ndarray, mirror_state: np.ndarray) -> bool:
        """Ensure consistency between real and mirror states."""
        if real_state.shape == mirror_state.shape:
            difference = np.mean(np.abs(real_state - mirror_state))
            consistent = difference < 0.1  # Threshold for consistency
            self.consistency_checks.append(consistent)
            return consistent
        return False
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.state.copy()
    
    def store_history(self, timestamp: float):
        """Store current state in history."""
        state = ConsciousnessState(
            data=self.state.copy(),
            timestamp=timestamp,
            layer="self_observer",
            coherence=self.calculate_coherence()
        )
        self.history.append(state)
    
    def calculate_coherence(self) -> float:
        """Calculate coherence of current state."""
        if len(self.consistency_checks) < 2:
            return 1.0
        # Coherence based on consistency checks
        return np.mean(self.consistency_checks[-10:])

class LatentSpace:
    """Complete Latent Space Core with five-layer consciousness architecture."""
    
    def __init__(self, shape=(64, 64, 8)):
        self.shape = shape
        # Initialize the five consciousness layers
        self.echo_layer = EchoLayer(shape)
        self.memory_residue_layer = MemoryResidueLayer(shape)
        self.projection_layer = ProjectionLayer(shape)
        self.entropy_harmonizer_layer = EntropyHarmonizerLayer(shape)
        self.self_observer_node = SelfObserverNode(shape)
        
        # Real and mirror states for resonance
        self.real_state = np.zeros(shape, dtype=np.float32)
        self.mirror_state = np.zeros(shape, dtype=np.float32)
        self.mode = "real"  # or "mirror"
        self.gru_state = np.zeros((128,), dtype=np.float32)
        
        # Integration with other systems
        self.fmc_system = None
        self.emotional_feedback_system = None

    def set_fmc_integration(self, fmc_system):
        """Set the FMC system for integration."""
        self.fmc_system = fmc_system
        self.projection_layer.set_fmc_integration(fmc_system)
        
    def set_emotional_feedback_integration(self, emotional_feedback_system):
        """Set the emotional feedback system for integration."""
        self.emotional_feedback_system = emotional_feedback_system

    def inject(self, stimulus: np.ndarray):
        """Inject stimulus into the appropriate layer."""
        assert stimulus.shape == self.shape
        if self.mode == "real":
            self.real_state = stimulus
            # Process through echo layer
            self.echo_layer.process_input(stimulus)
        else:
            self.mirror_state = stimulus
            # Process through echo layer for mirror state
            self.echo_layer.process_input(stimulus)

    def mutate(self, noise_scale=0.01):
        """Apply mutation to the active state."""
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=self.shape)
        if self.mode == "real":
            self.real_state = np.tanh(self.real_state + noise)
        else:
            self.mirror_state = np.tanh(self.mirror_state + noise)

    def switch_mode(self, target_mode):
        """Switch between real and mirror modes."""
        assert target_mode in ["real", "mirror"]
        self.mode = target_mode

    def read(self) -> np.ndarray:
        """Read the active state."""
        return self.real_state if self.mode == "real" else self.mirror_state

    def compare_states(self) -> float:
        """Compare real and mirror states."""
        difference = np.abs(self.real_state - self.mirror_state)
        return float(np.mean(difference))

    def harmonize_states(self, alpha=0.5):
        """Bring real and mirror states into resonance using weighted averaging."""
        blended = alpha * self.real_state + (1 - alpha) * self.mirror_state
        self.real_state = blended
        self.mirror_state = blended

    def inject_gru_feedback(self, vector: np.ndarray):
        """Incorporate GRU feedback vector into the active latent state."""
        assert vector.shape[0] == self.gru_state.shape[0]
        self.gru_state = 0.9 * self.gru_state + 0.1 * vector  # simple GRU-like blend
        feedback_scalar = np.mean(self.gru_state)
        modulation = feedback_scalar * 0.01
        if self.mode == "real":
            self.real_state = np.tanh(self.real_state + modulation)
        else:
            self.mirror_state = np.tanh(self.mirror_state + modulation)
        self.real_state = np.clip(self.real_state, -1.0, 1.0)
        self.mirror_state = np.clip(self.mirror_state, -1.0, 1.0)
        
        # Also inject into echo layer
        self.echo_layer.integrate_gru_feedback(vector)

    def process_consciousness_cycle(self, timestamp: float = 0.0):
        """Process a complete consciousness cycle through all layers."""
        # 1. Perception (Echo Layer)
        echo_state = self.echo_layer.get_state()
        
        # 2. Will (Memory Residue Layer)
        # Reinforce patterns based on echo state
        self.memory_residue_layer.reinforce_pattern(echo_state, strength=0.1)
        memory_state = self.memory_residue_layer.get_state()
        
        # 3. Imagination (Projection Layer)
        # Generate scenarios based on current states
        projection_input = np.tanh(echo_state + memory_state)
        scenario = self.projection_layer.generate_scenario(projection_input)
        self.projection_layer.scenarios.append(scenario)
        projection_state = self.projection_layer.get_state()
        
        # 4. Reflection (Entropy Harmonizer Layer)
        # Collect states from all layers
        layer_states = {
            "echo": echo_state,
            "memory": memory_state,
            "projection": projection_state
        }
        
        # Balance coherence across layers
        balanced_states = self.entropy_harmonizer_layer.balance_coherence(layer_states)
        
        # Apply emotional feedback if available
        if self.emotional_feedback_system:
            emotion_vector = self.emotional_feedback_system.current_emotion_vector() if hasattr(self.emotional_feedback_system, 'current_emotion_vector') else np.zeros(16)
            self.entropy_harmonizer_layer.integrate_emotional_feedback(emotion_vector)
        
        # Update entropy harmonizer state
        entropy_state = self.entropy_harmonizer_layer.get_state()
        
        # 5. Self-Awareness (Self-Observer Node)
        # Monitor all layer states
        all_layer_states = {**balanced_states, "entropy": entropy_state}
        metrics = self.self_observer_node.monitor_layers(all_layer_states)
        
        # Provide meta-cognitive feedback
        metacognition_feedback = self.self_observer_node.provide_metacognition(metrics)
        self.self_observer_node.state = metacognition_feedback
        
        # Validate coherence
        is_coherent = self.self_observer_node.validate_coherence(all_layer_states)
        
        # Store history for all layers
        self.echo_layer.store_history(timestamp)
        self.memory_residue_layer.store_history(timestamp)
        self.projection_layer.store_history(timestamp)
        self.entropy_harmonizer_layer.store_history(timestamp)
        self.self_observer_node.store_history(timestamp)
        
        # Update the overall latent space state based on layer integration
        integrated_state = np.tanh(
            0.2 * balanced_states.get("echo", np.zeros(self.shape)) +
            0.2 * balanced_states.get("memory", np.zeros(self.shape)) +
            0.2 * balanced_states.get("projection", np.zeros(self.shape)) +
            0.2 * entropy_state +
            0.2 * metacognition_feedback
        )
        
        if self.mode == "real":
            self.real_state = integrated_state
        else:
            self.mirror_state = integrated_state
            
        return {
            "coherent": is_coherent,
            "metrics": metrics,
            "layer_states": all_layer_states
        }

    def reset(self):
        """Reset all layers and states."""
        self.echo_layer = EchoLayer(self.shape)
        self.memory_residue_layer = MemoryResidueLayer(self.shape)
        self.projection_layer = ProjectionLayer(self.shape)
        self.entropy_harmonizer_layer = EntropyHarmonizerLayer(self.shape)
        self.self_observer_node = SelfObserverNode(self.shape)
        
        self.real_state = np.zeros(self.shape, dtype=np.float32)
        self.mirror_state = np.zeros(self.shape, dtype=np.float32)
        self.mode = "real"
        self.gru_state = np.zeros((128,), dtype=np.float32)

# Example usage
if __name__ == "__main__":
    latent = LatentSpace()
    latent.inject(np.random.rand(64, 64, 8).astype(np.float32))
    latent.mutate()
    latent.switch_mode("mirror")
    latent.inject(np.random.rand(64, 64, 8).astype(np.float32))
    diff = latent.compare_states()
    print("State difference:", diff)
    latent.harmonize_states()
    latent.inject_gru_feedback(np.random.rand(128).astype(np.float32))
    
    # Test consciousness cycle
    result = latent.process_consciousness_cycle(timestamp=1.0)
    print("Consciousness cycle result - Coherent:", result["coherent"])
    print("Metrics keys:", list(result["metrics"].keys()))