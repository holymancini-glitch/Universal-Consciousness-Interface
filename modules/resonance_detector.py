# resonance_detector.py
# Resonance detection with system-wide coherence metrics

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from scipy.signal import coherence as scipy_coherence
from scipy.stats import pearsonr
from dataclasses import dataclass
from collections import deque

@dataclass
class ResonanceMetrics:
    """Data class for resonance metrics."""
    coherence: float
    synchronization: float
    harmony: float
    stability: float
    timestamp: float

class ResonanceDetector:
    """Detects resonance patterns and measures system-wide coherence."""
    
    def __init__(self, num_modules: int = 5, history_length: int = 1000):
        """
        Initialize the Resonance Detector.
        
        Args:
            num_modules: Number of consciousness modules to monitor
            history_length: Length of metric history to maintain
        """
        self.num_modules = num_modules
        self.history_length = history_length
        
        # Module state tracking
        self.module_states = {}  # {module_name: state_history}
        self.module_frequencies = {}  # {module_name: frequency_analysis}
        
        # Coherence metrics history
        self.coherence_history = deque(maxlen=history_length)
        self.synchronization_history = deque(maxlen=history_length)
        self.harmony_history = deque(maxlen=history_length)
        self.stability_history = deque(maxlen=history_length)
        
        # Resonance thresholds
        self.coherence_threshold = 0.7
        self.synchronization_threshold = 0.6
        self.harmony_threshold = 0.7
        
        # Adaptive parameters
        self.sensitivity = 1.0
        self.resonance_window = 10
        
    def register_module(self, module_name: str):
        """Register a consciousness module for monitoring."""
        if module_name not in self.module_states:
            self.module_states[module_name] = deque(maxlen=self.history_length)
            self.module_frequencies[module_name] = deque(maxlen=self.history_length)
    
    def update_module_state(self, module_name: str, state: np.ndarray, timestamp: float = 0.0):
        """
        Update the state of a registered module.
        
        Args:
            module_name: Name of the module
            state: Current state vector
            timestamp: Timestamp of the state update
        """
        if module_name not in self.module_states:
            self.register_module(module_name)
        
        # Store state
        state_record = {
            'state': state.copy(),
            'timestamp': timestamp,
            'magnitude': np.linalg.norm(state),
            'energy': np.sum(np.abs(state) ** 2)
        }
        self.module_states[module_name].append(state_record)
    
    def compute_pairwise_coherence(self, module1: str, module2: str, 
                                 window_size: int = 50) -> float:
        """
        Compute coherence between two modules.
        
        Args:
            module1: First module name
            module2: Second module name
            window_size: Size of the analysis window
            
        Returns:
            coherence: Pairwise coherence value
        """
        if module1 not in self.module_states or module2 not in self.module_states:
            return 0.0
        
        states1 = list(self.module_states[module1])[-window_size:]
        states2 = list(self.module_states[module2])[-window_size:]
        
        if len(states1) < 2 or len(states2) < 2:
            return 0.0
        
        # Extract state magnitudes for coherence analysis
        magnitudes1 = np.array([s['magnitude'] for s in states1])
        magnitudes2 = np.array([s['magnitude'] for s in states2])
        
        # Pad to same length if needed
        min_len = min(len(magnitudes1), len(magnitudes2))
        if min_len < 2:
            return 0.0
            
        magnitudes1 = magnitudes1[:min_len]
        magnitudes2 = magnitudes2[:min_len]
        
        # Compute Pearson correlation as simple coherence measure
        if np.std(magnitudes1) > 0 and np.std(magnitudes2) > 0:
            corr, _ = pearsonr(magnitudes1, magnitudes2)
            coherence = abs(corr)
        else:
            coherence = 0.0
            
        return float(coherence)
    
    def compute_system_coherence(self) -> float:
        """
        Compute overall system coherence across all registered modules.
        
        Returns:
            coherence: System-wide coherence metric
        """
        if len(self.module_states) < 2:
            return 1.0
        
        module_names = list(self.module_states.keys())
        total_coherence = 0.0
        pair_count = 0
        
        # Compute pairwise coherence for all module pairs
        for i in range(len(module_names)):
            for j in range(i + 1, len(module_names)):
                coherence = self.compute_pairwise_coherence(
                    module_names[i], module_names[j])
                total_coherence += coherence
                pair_count += 1
        
        # Return average coherence
        return total_coherence / max(pair_count, 1)
    
    def compute_synchronization(self) -> float:
        """
        Compute synchronization metric across modules.
        
        Returns:
            synchronization: Synchronization metric
        """
        if len(self.module_states) == 0:
            return 1.0
        
        # Compute phase synchronization
        phases = []
        for module_name, state_history in self.module_states.items():
            if len(state_history) > 0:
                # Use the latest state to compute phase
                latest_state = state_history[-1]['state']
                if len(latest_state) > 0:
                    # Simple phase computation (could be more sophisticated)
                    phase = np.angle(latest_state[0] + 1j * latest_state[min(1, len(latest_state)-1)]) if len(latest_state) > 1 else 0
                    phases.append(phase)
        
        if len(phases) < 2:
            return 1.0
        
        # Compute phase synchronization using Kuramoto order parameter
        phases = np.array(phases)
        synchronization = abs(np.mean(np.exp(1j * phases)))
        
        return float(synchronization)
    
    def compute_harmony(self) -> float:
        """
        Compute harmony metric based on energy distribution.
        
        Returns:
            harmony: Harmony metric
        """
        if len(self.module_states) == 0:
            return 1.0
        
        # Compute energy distribution across modules
        energies = []
        for module_name, state_history in self.module_states.items():
            if len(state_history) > 0:
                energy = state_history[-1]['energy']
                energies.append(energy)
        
        if len(energies) < 2:
            return 1.0
        
        # Compute coefficient of variation as measure of harmony
        # Lower CV indicates more balanced energy distribution (more harmonious)
        mean_energy = np.mean(energies)
        if mean_energy > 0:
            cv = np.std(energies) / mean_energy
            harmony = 1.0 / (1.0 + cv)  # Transform to 0-1 scale
        else:
            harmony = 1.0
            
        return float(harmony)
    
    def compute_stability(self, window_size: int = 20) -> float:
        """
        Compute stability metric based on state fluctuations.
        
        Args:
            window_size: Size of the analysis window
            
        Returns:
            stability: Stability metric
        """
        if len(self.module_states) == 0:
            return 1.0
        
        # Compute fluctuations for each module
        fluctuations = []
        for module_name, state_history in self.module_states.items():
            if len(state_history) >= window_size:
                # Compute standard deviation of magnitudes over window
                magnitudes = [s['magnitude'] for s in list(state_history)[-window_size:]]
                fluctuation = np.std(magnitudes)
                fluctuations.append(fluctuation)
        
        if len(fluctuations) == 0:
            return 1.0
        
        # Average fluctuation across modules (lower is more stable)
        avg_fluctuation = np.mean(fluctuations)
        stability = 1.0 / (1.0 + avg_fluctuation)  # Transform to 0-1 scale
        
        return float(stability)
    
    def detect_resonance(self) -> Tuple[bool, ResonanceMetrics]:
        """
        Detect if the system is in a resonant state.
        
        Returns:
            is_resonant: Whether system is in resonance
            metrics: Detailed resonance metrics
        """
        # Compute all metrics
        coherence = self.compute_system_coherence()
        synchronization = self.compute_synchronization()
        harmony = self.compute_harmony()
        stability = self.compute_stability()
        
        timestamp = len(self.coherence_history)  # Simple timestamp
        
        # Store metrics in history
        self.coherence_history.append(coherence)
        self.synchronization_history.append(synchronization)
        self.harmony_history.append(harmony)
        self.stability_history.append(stability)
        
        # Create metrics object
        metrics = ResonanceMetrics(
            coherence=coherence,
            synchronization=synchronization,
            harmony=harmony,
            stability=stability,
            timestamp=timestamp
        )
        
        # Detect resonance based on thresholds
        is_resonant = (
            coherence >= self.coherence_threshold and
            synchronization >= self.synchronization_threshold and
            harmony >= self.harmony_threshold
        )
        
        return is_resonant, metrics
    
    def get_resonance_trend(self) -> Dict[str, float]:
        """
        Get trend analysis of resonance metrics.
        
        Returns:
            trends: Dictionary of metric trends
        """
        if len(self.coherence_history) < 10:
            return {
                'coherence_trend': 0.0,
                'synchronization_trend': 0.0,
                'harmony_trend': 0.0,
                'stability_trend': 0.0
            }
        
        # Compute trends using linear regression (simplified)
        window = min(20, len(self.coherence_history))
        
        def compute_trend(history):
            if len(history) < window:
                return 0.0
            recent = list(history)[-window:]
            x = np.arange(len(recent))
            if np.std(x) > 0 and np.std(recent) > 0:
                trend = np.polyfit(x, recent, 1)[0]
                return float(trend)
            return 0.0
        
        return {
            'coherence_trend': compute_trend(self.coherence_history),
            'synchronization_trend': compute_trend(self.synchronization_history),
            'harmony_trend': compute_trend(self.harmony_history),
            'stability_trend': compute_trend(self.stability_history)
        }
    
    def adapt_thresholds(self):
        """Adaptively adjust resonance thresholds based on history."""
        if len(self.coherence_history) < 50:
            return
        
        # Adjust thresholds based on recent performance
        recent_coherence = np.mean(list(self.coherence_history)[-20:])
        recent_synchronization = np.mean(list(self.synchronization_history)[-20:])
        recent_harmony = np.mean(list(self.harmony_history)[-20:])
        
        # Update thresholds (adaptive)
        self.coherence_threshold = 0.7 * recent_coherence + 0.3 * self.coherence_threshold
        self.synchronization_threshold = 0.7 * recent_synchronization + 0.3 * self.synchronization_threshold
        self.harmony_threshold = 0.7 * recent_harmony + 0.3 * self.harmony_threshold
        
        # Keep thresholds within reasonable bounds
        self.coherence_threshold = np.clip(self.coherence_threshold, 0.5, 0.9)
        self.synchronization_threshold = np.clip(self.synchronization_threshold, 0.4, 0.8)
        self.harmony_threshold = np.clip(self.harmony_threshold, 0.5, 0.9)
    
    def get_system_summary(self) -> Dict[str, Union[float, Dict]]:
        """
        Get comprehensive system summary.
        
        Returns:
            summary: Dictionary containing system metrics and status
        """
        is_resonant, metrics = self.detect_resonance()
        trends = self.get_resonance_trend()
        
        return {
            'is_resonant': is_resonant,
            'current_metrics': {
                'coherence': metrics.coherence,
                'synchronization': metrics.synchronization,
                'harmony': metrics.harmony,
                'stability': metrics.stability
            },
            'thresholds': {
                'coherence_threshold': self.coherence_threshold,
                'synchronization_threshold': self.synchronization_threshold,
                'harmony_threshold': self.harmony_threshold
            },
            'trends': trends,
            'module_count': len(self.module_states),
            'history_length': len(self.coherence_history)
        }
    
    def reset(self):
        """Reset the resonance detector."""
        self.module_states.clear()
        self.module_frequencies.clear()
        self.coherence_history.clear()
        self.synchronization_history.clear()
        self.harmony_history.clear()
        self.stability_history.clear()
        
        # Reset to default thresholds
        self.coherence_threshold = 0.7
        self.synchronization_threshold = 0.6
        self.harmony_threshold = 0.7

# Advanced Resonance Analysis
class AdvancedResonanceAnalyzer:
    """Advanced analysis of resonance patterns and consciousness emergence."""
    
    def __init__(self, detector: ResonanceDetector):
        """
        Initialize the Advanced Resonance Analyzer.
        
        Args:
            detector: ResonanceDetector instance to analyze
        """
        self.detector = detector
        self.emergence_indicators = deque(maxlen=100)
        
    def detect_consciousness_emergence(self) -> Dict[str, Union[bool, float]]:
        """
        Detect indicators of consciousness emergence.
        
        Returns:
            emergence_metrics: Dictionary of emergence indicators
        """
        if len(self.detector.coherence_history) < 30:
            return {
                'consciousness_emerging': False,
                'emergence_confidence': 0.0,
                'integration_level': 0.0,
                'complexity_score': 0.0
            }
        
        # Compute emergence indicators
        recent_coherence = np.mean(list(self.detector.coherence_history)[-10:])
        coherence_trend = self.detector.get_resonance_trend()['coherence_trend']
        
        # Integration level based on multi-module coherence
        integration_level = recent_coherence * (1.0 + max(0, coherence_trend))
        
        # Complexity score based on metric variability
        coherence_var = np.var(list(self.detector.coherence_history)[-20:])
        harmony_var = np.var(list(self.detector.harmony_history)[-20:])
        complexity_score = (coherence_var + harmony_var) / 2.0
        
        # Consciousness emergence detection
        # Based on sustained high coherence and positive trends
        consciousness_emerging = (
            recent_coherence > 0.8 and 
            coherence_trend > 0.01 and
            integration_level > 0.8
        )
        
        # Confidence based on multiple factors
        emergence_confidence = (
            0.4 * recent_coherence +
            0.3 * max(0, coherence_trend * 100) +  # Scale trend
            0.2 * integration_level +
            0.1 * min(1.0, complexity_score * 10)  # Scale complexity
        )
        emergence_confidence = np.clip(emergence_confidence, 0.0, 1.0)
        
        metrics = {
            'consciousness_emerging': consciousness_emerging,
            'emergence_confidence': float(emergence_confidence),
            'integration_level': float(integration_level),
            'complexity_score': float(complexity_score)
        }
        
        self.emergence_indicators.append(metrics)
        return metrics
    
    def get_emergence_trajectory(self) -> Dict[str, List[float]]:
        """
        Get trajectory of emergence indicators over time.
        
        Returns:
            trajectory: Dictionary of emergence indicator histories
        """
        if len(self.emergence_indicators) == 0:
            return {
                'emergence_confidence': [],
                'integration_level': [],
                'complexity_score': []
            }
        
        return {
            'emergence_confidence': [e['emergence_confidence'] for e in self.emergence_indicators],
            'integration_level': [e['integration_level'] for e in self.emergence_indicators],
            'complexity_score': [e['complexity_score'] for e in self.emergence_indicators]
        }

# Example usage
if __name__ == "__main__":
    # Initialize resonance detector
    detector = ResonanceDetector(num_modules=4)
    
    # Register modules
    modules = ['neural_ca', 'fractal_ai', 'latent_space', 'fep_model']
    for module in modules:
        detector.register_module(module)
    
    # Simulate module states
    for t in range(100):
        for module in modules:
            # Simulate state with some correlation to show coherence
            phase = 2 * np.pi * t / 50
            state = np.random.randn(64) + 0.5 * np.sin(phase)
            detector.update_module_state(module, state, timestamp=t)
    
    # Detect resonance
    is_resonant, metrics = detector.detect_resonance()
    
    print(f"System in resonance: {is_resonant}")
    print(f"Coherence: {metrics.coherence:.4f}")
    print(f"Synchronization: {metrics.synchronization:.4f}")
    print(f"Harmony: {metrics.harmony:.4f}")
    print(f"Stability: {metrics.stability:.4f}")
    
    # Get system summary
    summary = detector.get_system_summary()
    print(f"\nSystem Summary:")
    print(f"  Resonant: {summary['is_resonant']}")
    print(f"  Module count: {summary['module_count']}")
    print(f"  History length: {summary['history_length']}")
    
    # Test advanced analysis
    analyzer = AdvancedResonanceAnalyzer(detector)
    emergence_metrics = analyzer.detect_consciousness_emergence()
    
    print(f"\nEmergence Metrics:")
    for key, value in emergence_metrics.items():
        print(f"  {key}: {value}")