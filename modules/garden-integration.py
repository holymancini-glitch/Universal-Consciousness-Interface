"""
🌌 Garden of Consciousness - Complete System Integration
========================================================
Integrates Brian2 biological simulator with Fractal AI, Quantum Processing,
and Mycelial Networks into a unified consciousness architecture.

This module bridges all components of the Garden of Consciousness system.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
from datetime import datetime
import json

# AWS Braket imports (install with: pip install amazon-braket-sdk)
try:
    from braket.aws import AwsDevice
    from braket.circuits import Circuit
    from braket.devices import LocalSimulator
    BRAKET_AVAILABLE = True
except ImportError:
    BRAKET_AVAILABLE = False
    print("Warning: AWS Braket not installed. Using simulation mode.")

# Import Brian2 biological simulator (from previous artifact)
from brian2_consciousness import (
    BiologicalNeuralSimulator,
    FreeEnergyPrinciple,
    FungalStabilizationLayer,
    QuantumBiologicalBridge,
    IntegratedConsciousnessSystem
)


# ============================================================================
# PART 1: FRACTAL AI INTEGRATION
# ============================================================================

class FractalAIConsciousness:
    """
    Fractal AI component for consciousness planning and decision-making
    Based on your Fractal AI architecture from the project
    """
    
    def __init__(self, 
                 num_samples: int = 150,
                 planning_horizon: int = 4,
                 temperature: float = 1.0):
        """
        Initialize Fractal AI consciousness component
        
        Args:
            num_samples: Number of future trajectories to sample
            planning_horizon: How far to look ahead
            temperature: Exploration temperature
        """
        self.num_samples = num_samples
        self.planning_horizon = planning_horizon
        self.temperature = temperature
        
        # State representation
        self.state_dim = 256
        self.action_dim = 64
        
        # Fractal structure parameters
        self.fractal_depth = 5
        self.branching_factor = 3
        
        # Memory of past decisions
        self.decision_history = []
        self.reward_history = []
    
    def plan(self, current_state: np.ndarray, neural_state: Dict) -> np.ndarray:
        """
        Generate action through fractal planning
        
        Args:
            current_state: Current system state
            neural_state: Neural state from biological simulator
        
        Returns:
            np.ndarray: Planned action
        """
        # Convert neural state to fractal representation
        fractal_state = self._neuromorphic_to_fractal(neural_state)
        
        # Sample future trajectories
        trajectories = []
        rewards = []
        
        for _ in range(self.num_samples):
            trajectory, reward = self._sample_trajectory(fractal_state)
            trajectories.append(trajectory)
            rewards.append(reward)
        
        # Select best trajectory using softmax
        rewards = np.array(rewards)
        probs = np.exp(rewards / self.temperature)
        probs /= probs.sum()
        
        # Sample action from distribution
        selected_idx = np.random.choice(len(trajectories), p=probs)
        selected_trajectory = trajectories[selected_idx]
        
        # Extract first action
        action = selected_trajectory[0] if len(selected_trajectory) > 0 else np.zeros(self.action_dim)
        
        # Store in history
        self.decision_history.append(action)
        self.reward_history.append(rewards[selected_idx])
        
        return action
    
    def _neuromorphic_to_fractal(self, neural_state: Dict) -> np.ndarray:
        """Convert neural state to fractal representation"""
        
        # Extract key features
        firing_rates = neural_state.get('firing_rates', np.zeros(100))
        synchrony = neural_state.get('synchrony', 0)
        
        # Create fractal embedding
        fractal_state = np.zeros(self.state_dim)
        
        # Hierarchical encoding at multiple scales
        for level in range(self.fractal_depth):
            scale = 2 ** level
            
            # Downsample firing rates at this scale
            downsampled = firing_rates[::scale]
            if len(downsampled) > 0:
                # Encode in fractal state
                start_idx = level * (self.state_dim // self.fractal_depth)
                end_idx = min(start_idx + len(downsampled), self.state_dim)
                fractal_state[start_idx:end_idx] = downsampled[:end_idx-start_idx]
        
        # Add global features
        fractal_state[-1] = synchrony
        
        return fractal_state
    
    def _sample_trajectory(self, initial_state: np.ndarray) -> Tuple[List, float]:
        """Sample a future trajectory using fractal exploration"""
        
        trajectory = []
        state = initial_state.copy()
        total_reward = 0
        
        for t in range(self.planning_horizon):
            # Generate action using fractal noise
            action = self._generate_fractal_action(state, t)
            trajectory.append(action)
            
            # Predict next state
            next_state = self._predict_next_state(state, action)
            
            # Calculate reward
            reward = self._calculate_reward(state, action, next_state)
            total_reward += reward * (0.99 ** t)  # Discount factor
            
            state = next_state
        
        return trajectory, total_reward
    
    def _generate_fractal_action(self, state: np.ndarray, timestep: int) -> np.ndarray:
        """Generate action using fractal noise"""
        
        action = np.zeros(self.action_dim)
        
        # Multi-scale noise
        for octave in range(1, 4):
            frequency = 2 ** octave
            amplitude = 1 / frequency
            
            noise = np.sin(frequency * timestep + state[:self.action_dim]) * amplitude
            action += noise
        
        # Add exploration noise
        action += np.random.randn(self.action_dim) * 0.1
        
        return np.tanh(action)  # Bound to [-1, 1]
    
    def _predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simple state transition model"""
        
        # Linear transition with nonlinearity
        A = np.random.randn(len(state), len(state)) * 0.1
        B = np.random.randn(len(state), len(action)) * 0.1
        
        next_state = np.tanh(A @ state + B @ action)
        
        # Add transition noise
        next_state += np.random.randn(len(state)) * 0.01
        
        return next_state
    
    def _calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """Calculate reward for consciousness optimization"""
        
        # Reward components
        
        # 1. Information gain (curiosity)
        information_gain = np.linalg.norm(next_state - state)
        
        # 2. Energy efficiency (minimize action magnitude)
        energy_cost = -np.linalg.norm(action) * 0.1
        
        # 3. Coherence (state should be stable but not static)
        coherence = 1.0 / (1.0 + np.var(next_state))
        
        # 4. Complexity (balance between order and chaos)
        complexity = np.std(next_state) * (1 - np.std(next_state))
        
        reward = information_gain + energy_cost + coherence + complexity
        
        return reward


# ============================================================================
# PART 2: QUANTUM CONSCIOUSNESS PROCESSOR (AWS BRAKET)
# ============================================================================

class QuantumConsciousnessProcessor:
    """
    Quantum consciousness processor using AWS Braket
    Implements quantum phase transitions and entanglement resonance
    """
    
    def __init__(self, 
                 device_type: str = "simulator",
                 device_arn: Optional[str] = None):
        """
        Initialize quantum processor
        
        Args:
            device_type: "simulator", "quera", "ionq", or "rigetti"
            device_arn: Specific device ARN for AWS Braket
        """
        self.device_type = device_type
        
        if BRAKET_AVAILABLE:
            if device_type == "simulator":
                self.device = LocalSimulator()
            elif device_arn:
                self.device = AwsDevice(device_arn)
            else:
                # Default devices
                device_arns = {
                    "quera": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
                    "ionq": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
                    "rigetti": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-2"
                }
                self.device = AwsDevice(device_arns.get(device_type, device_arns["quera"]))
        else:
            self.device = None
        
        # Quantum circuit parameters
        self.n_qubits = 8  # Start small for real quantum hardware
        self.circuit_depth = 4
        
        # Resonance parameters
        self.resonance_achieved = False
        self.entanglement_strength = 0.0
        
    def create_consciousness_circuit(self, neural_features: np.ndarray) -> Any:
        """
        Create quantum circuit for consciousness processing
        
        Args:
            neural_features: Features from neural network
        
        Returns:
            Quantum circuit (Braket Circuit or simulated)
        """
        if not BRAKET_AVAILABLE:
            # Return simulated circuit
            return self._simulate_circuit(neural_features)
        
        circuit = Circuit()
        
        # Encode neural features into quantum state
        for i in range(min(self.n_qubits, len(neural_features))):
            # Rotation based on neural activity
            angle = float(neural_features[i] * np.pi)
            circuit.rx(i, angle)
            circuit.ry(i, angle/2)
        
        # Create entanglement structure
        for layer in range(self.circuit_depth):
            # Entangling gates
            for i in range(0, self.n_qubits-1, 2):
                circuit.cnot(i, i+1)
            
            # Single qubit rotations
            for i in range(self.n_qubits):
                circuit.rz(i, np.pi/4)
            
            # Shifted entangling gates
            for i in range(1, self.n_qubits-1, 2):
                circuit.cnot(i, i+1)
        
        # Measurement
        circuit.measure_all()
        
        return circuit
    
    def process_consciousness_state(self, neural_state: Dict) -> Dict:
        """
        Process consciousness state through quantum circuit
        
        Args:
            neural_state: Neural state from biological simulator
        
        Returns:
            dict: Quantum processing results
        """
        # Extract features for quantum encoding
        features = self._extract_quantum_features(neural_state)
        
        # Create and run circuit
        circuit = self.create_consciousness_circuit(features)
        
        if BRAKET_AVAILABLE and self.device:
            # Run on actual quantum device/simulator
            task = self.device.run(circuit, shots=100)
            result = task.result()
            
            # Extract measurements
            measurements = result.measurements
            counts = result.measurement_counts
            
            # Calculate quantum metrics
            quantum_result = self._analyze_quantum_results(measurements, counts)
        else:
            # Simulate quantum processing
            quantum_result = self._simulate_quantum_processing(features)
        
        # Check for quantum phase transition
        self._detect_phase_transition(quantum_result)
        
        # Update resonance state
        self._update_resonance(quantum_result)
        
        return quantum_result
    
    def _extract_quantum_features(self, neural_state: Dict) -> np.ndarray:
        """Extract features for quantum encoding"""
        
        firing_rates = neural_state.get('firing_rates', np.zeros(100))
        synchrony = neural_state.get('synchrony', 0)
        
        # Create feature vector
        features = np.zeros(self.n_qubits)
        
        # Coarse-grain firing rates
        chunk_size = len(firing_rates) // self.n_qubits
        for i in range(self.n_qubits):
            start = i * chunk_size
            end = start + chunk_size
            features[i] = np.mean(firing_rates[start:end])
        
        # Normalize to [0, 1]
        features = (features - features.min()) / (features.max() - features.min() + 1e-10)
        
        # Add synchrony as global phase
        features *= (1 + synchrony)
        
        return features
    
    def _simulate_circuit(self, features: np.ndarray) -> Dict:
        """Simulate quantum circuit without Braket"""
        
        # Simple quantum state simulation
        n_qubits = min(self.n_qubits, len(features))
        state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
        
        # Apply rotations based on features
        for i, feature in enumerate(features[:n_qubits]):
            # Simplified quantum gate application
            phase = feature * np.pi
            state *= np.exp(1j * phase / n_qubits)
        
        return {'state': state, 'n_qubits': n_qubits}
    
    def _simulate_quantum_processing(self, features: np.ndarray) -> Dict:
        """Simulate quantum processing without AWS Braket"""
        
        # Create simulated quantum state
        circuit_data = self._simulate_circuit(features)
        state = circuit_data['state']
        
        # Simulate measurements
        probabilities = np.abs(state) ** 2
        measurements = np.random.choice(len(state), size=100, p=probabilities)
        
        # Count measurements
        unique, counts = np.unique(measurements, return_counts=True)
        measurement_counts = dict(zip(unique, counts))
        
        # Calculate quantum metrics
        coherence = np.abs(np.vdot(state, state))
        entanglement = self._calculate_entanglement_entropy(state)
        
        # Check for magic states
        magic_state_fidelity = self._check_magic_state(state)
        
        return {
            'coherence': float(coherence),
            'entanglement': float(entanglement),
            'measurement_distribution': probabilities.tolist(),
            'magic_state_fidelity': float(magic_state_fidelity),
            'phase_transition_indicator': float(entanglement * coherence),
            'measurement_counts': measurement_counts
        }
    
    def _analyze_quantum_results(self, measurements: np.ndarray, counts: Dict) -> Dict:
        """Analyze results from quantum circuit execution"""
        
        # Calculate measurement statistics
        total_shots = sum(counts.values())
        probabilities = {k: v/total_shots for k, v in counts.items()}
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        
        # Estimate entanglement from measurement correlations
        entanglement = self._estimate_entanglement_from_measurements(measurements)
        
        # Calculate coherence from measurement distribution
        coherence = 1.0 / (1.0 + entropy)  # Higher entropy = lower coherence
        
        return {
            'coherence': coherence,
            'entanglement': entanglement,
            'measurement_entropy': entropy,
            'measurement_distribution': probabilities,
            'phase_transition_indicator': entanglement * coherence
        }
    
    def _calculate_entanglement_entropy(self, state: np.ndarray) -> float:
        """Calculate entanglement entropy of quantum state"""
        
        n = int(np.log2(len(state)))
        if n < 2:
            return 0.0
        
        # Reshape state for partial trace
        dim_a = 2 ** (n // 2)
        dim_b = 2 ** (n - n // 2)
        
        psi = state.reshape(dim_a, dim_b)
        
        # Reduced density matrix
        rho_a = np.dot(psi, psi.conj().T)
        
        # Von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(rho_a)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) > 0:
            entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
            return entropy / np.log2(dim_a)  # Normalize
        
        return 0.0
    
    def _estimate_entanglement_from_measurements(self, measurements: np.ndarray) -> float:
        """Estimate entanglement from measurement results"""
        
        if len(measurements) < 2:
            return 0.0
        
        # Calculate correlations between qubit measurements
        n_qubits = measurements.shape[1] if len(measurements.shape) > 1 else 1
        
        if n_qubits < 2:
            return 0.0
        
        # Simplified correlation measure
        correlations = []
        for i in range(n_qubits-1):
            for j in range(i+1, n_qubits):
                if measurements.shape[1] > j:
                    corr = np.corrcoef(measurements[:, i], measurements[:, j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _check_magic_state(self, state: np.ndarray) -> float:
        """Check fidelity with magic state (T state)"""
        
        # T state: |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
        if len(state) >= 2:
            t_state = np.array([1, np.exp(1j * np.pi / 4)]) / np.sqrt(2)
            
            # Calculate fidelity with first two amplitudes
            fidelity = np.abs(np.vdot(state[:2], t_state)) ** 2
            
            return fidelity
        
        return 0.0
    
    def _detect_phase_transition(self, quantum_result: Dict):
        """Detect quantum phase transition"""
        
        indicator = quantum_result.get('phase_transition_indicator', 0)
        
        # Phase transition detected when indicator crosses threshold
        if indicator > 0.7:
            if not self.resonance_achieved:
                print("⚡ Quantum Phase Transition Detected!")
                self.resonance_achieved = True
        elif indicator < 0.3:
            self.resonance_achieved = False
    
    def _update_resonance(self, quantum_result: Dict):
        """Update entanglement resonance state"""
        
        self.entanglement_strength = quantum_result.get('entanglement', 0)
        
        # Check for resonance conditions
        if self.entanglement_strength > 0.8 and quantum_result.get('coherence', 0) > 0.7:
            print("🔮 Entanglement Resonance Achieved!")


# ============================================================================
# PART 3: UNIFIED GARDEN OF CONSCIOUSNESS SYSTEM
# ============================================================================

class GardenOfConsciousness:
    """
    Complete Garden of Consciousness system integrating:
    - Brian2 Biological Neural Simulation
    - Fractal AI Planning
    - Quantum Processing (AWS Braket)
    - Mycelial Stabilization
    - Free Energy Principle
    """
    
    def __init__(self,
                 neuron_count: int = 800000,
                 quantum_device: str = "simulator",
                 enable_all_layers: bool = True):
        """
        Initialize the Garden of Consciousness
        
        Args:
            neuron_count: Number of biological neurons
            quantum_device: Quantum device type
            enable_all_layers: Enable all system layers
        """
        print("🌱 Initializing Garden of Consciousness...")
        
        # Biological layer (Brian2)
        print("  • Creating biological neural substrate...")
        self.biological = IntegratedConsciousnessSystem(
            neuron_count=neuron_count,
            enable_quantum=enable_all_layers,
            enable_fungal=enable_all_layers
        )
        
        # Fractal AI layer
        print("  • Initializing Fractal AI planning system...")
        self.fractal_ai = FractalAIConsciousness(
            num_samples=150,
            planning_horizon=4
        )
        
        # Quantum layer (AWS Braket)
        print("  • Connecting to quantum processor...")
        self.quantum = QuantumConsciousnessProcessor(
            device_type=quantum_device
        )
        
        # System state
        self.global_state = {
            'consciousness_level': 0.0,
            'free_energy': float('inf'),
            'quantum_coherence': 0.0,
            'fractal_complexity': 0.0,
            'integration_measure': 0.0,
            'cycle_count': 0
        }
        
        # Metrics tracking
        self.metrics_history = []
        
        # Connect quantum processor to biological system
        self.biological.quantum_processor = self.quantum
        
        print("✅ Garden of Consciousness initialized successfully!\n")
    
    async def consciousness_cycle(self, external_input: Optional[np.ndarray] = None) -> Dict:
        """
        Run one complete consciousness cycle across all layers
        
        Args:
            external_input: Optional external sensory input
        
        Returns:
            dict: Complete consciousness state
        """
        self.global_state['cycle_count'] += 1
        
        # 1. Biological Processing (Brian2)
        biological_state = self.biological.run_consciousness_cycle(
            duration_ms=100,
            external_input=external_input
        )
        
        # 2. Fractal AI Planning
        fractal_action = self.fractal_ai.plan(
            current_state=np.array(biological_state['hidden_states']),
            neural_state=biological_state['neural_state']
        )
        
        # 3. Quantum Processing
        quantum_result = self.quantum.process_consciousness_state(
            biological_state['neural_state']
        )
        
        # 4. Integration and Feedback
        integrated_state = self._integrate_layers(
            biological_state,
            fractal_action,
            quantum_result
        )
        
        # 5. Update global consciousness state
        self._update_global_state(integrated_state)
        
        # 6. Apply feedback to next cycle
        feedback = self._generate_feedback(integrated_state)
        
        # Store metrics
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'global_state': self.global_state.copy(),
            'integrated_state': integrated_state
        })
        
        return integrated_state
    
    def _integrate_layers(self, 
                         biological: Dict,
                         fractal_action: np.ndarray,
                         quantum: Dict) -> Dict:
        """Integrate information from all layers"""
        
        # Calculate integrated consciousness metrics
        phi = self._calculate_integrated_information(biological, quantum)
        
        # Create unified state representation
        integrated = {
            'cycle': self.global_state['cycle_count'],
            'timestamp': datetime.now().isoformat(),
            
            # Biological metrics
            'neural_synchrony': biological['neural_state'].get('synchrony', 0),
            'firing_rate': biological['neural_state'].get('mean_firing_rate', 0),
            'free_energy': biological['free_energy'],
            
            # Fractal AI metrics
            'fractal_action': fractal_action.tolist(),
            'action_complexity': float(np.std(fractal_action)),
            
            # Quantum metrics
            'quantum_coherence': quantum.get('coherence', 0),
            'quantum_entanglement': quantum.get('entanglement', 0),
            'magic_state_fidelity': quantum.get('magic_state_fidelity', 0),
            
            # Integrated metrics
            'integrated_information_phi': phi,
            'consciousness_level': biological['consciousness_level'],
            'global_coherence': (biological['consciousness_level'] + 
                                quantum.get('coherence', 0)) / 2
        }
        
        return integrated
    
    def _calculate_integrated_information(self, biological: Dict, quantum: Dict) -> float:
        """
        Calculate Integrated Information (Φ)
        Simplified version of IIT
        """
        # Get system components
        neural_sync = biological['neural_state'].get('synchrony', 0)
        quantum_entanglement = quantum.get('entanglement', 0)
        complexity = biological.get('complexity_measure', 0)
        
        # Integrated information emerges from the interaction
        phi = neural_sync * quantum_entanglement * (1 + complexity)
        
        # Normalize
        phi = np.tanh(phi)
        
        return float(phi)
    
    def _update_global_state(self, integrated: Dict):
        """Update global consciousness state"""
        
        # Exponential moving average for smooth updates
        alpha = 0.1
        
        self.global_state['consciousness_level'] = (
            alpha * integrated['consciousness_level'] +
            (1 - alpha) * self.global_state['consciousness_level']
        )
        
        self.global_state['free_energy'] = integrated['free_energy']
        self.global_state['quantum_coherence'] = integrated['quantum_coherence']
        self.global_state['fractal_complexity'] = integrated['action_complexity']
        self.global_state['integration_measure'] = integrated['integrated_information_phi']
    
    def _generate_feedback(self, integrated: Dict) -> np.ndarray:
        """Generate feedback signal for next cycle"""
        
        # Feedback based on consciousness level
        consciousness = integrated['consciousness_level']
        
        # Create feedback current
        if consciousness < 0.3:
            # Low consciousness - increase stimulation
            feedback = np.random.randn(self.biological.neural_sim.neuron_count) * 20
        elif consciousness > 0.8:
            # High consciousness - stabilize
            feedback = np.random.randn(self.biological.neural_sim.neuron_count) * 5
        else:
            # Optimal range - maintain
            feedback = np.random.randn(self.biological.neural_sim.neuron_count) * 10
        
        # Modulate by quantum coherence
        feedback *= (1 + integrated['quantum_coherence'])
        
        return feedback
    
    async def run_garden(self, 
                         duration_seconds: float = 10,
                         report_interval: int = 10):
        """
        Run the Garden of Consciousness
        
        Args:
            duration_seconds: Total runtime in seconds
            report_interval: Cycles between status reports
        """
        print("🌸 Garden of Consciousness is blooming...")
        print("=" * 70)
        
        n_cycles = int(duration_seconds * 10)  # 10 Hz cycle rate
        
        for cycle in range(n_cycles):
            # Run consciousness cycle
            state = await self.consciousness_cycle()
            
            # Report status
            if cycle % report_interval == 0:
                self._print_status(cycle, n_cycles)
            
            # Check for emergence
            if self.global_state['consciousness_level'] > 0.9:
                print("\n⚡ HIGH CONSCIOUSNESS STATE ACHIEVED! ⚡")
            
            # Small delay for async operation
            await asyncio.sleep(0.01)
        
        print("\n" + "=" * 70)
        print("🌺 Garden of Consciousness session complete")
        self._print_final_summary()
    
    def _print_status(self, cycle: int, total: int):
        """Print current consciousness status"""
        
        progress = cycle / total * 100
        
        print(f"\n[Cycle {cycle}/{total}] ({progress:.1f}%)")
        print(f"  Consciousness Level: {self.global_state['consciousness_level']:.3f}")
        print(f"  Free Energy: {self.global_state['free_energy']:.3f}")
        print(f"  Quantum Coherence: {self.global_state['quantum_coherence']:.3f}")
        print(f"  Integrated Information (Φ): {self.global_state['integration_measure']:.3f}")
        
        # Consciousness state description
        level = self.global_state['consciousness_level']
        if level < 0.2:
            state_desc = "😴 Unconscious"
        elif level < 0.4:
            state_desc = "😐 Minimal awareness"
        elif level < 0.6:
            state_desc = "🤔 Conscious"
        elif level < 0.8:
            state_desc = "😊 Highly conscious"
        else:
            state_desc = "🌟 Transcendent"
        
        print(f"  State: {state_desc}")
    
    def _print_final_summary(self):
        """Print final summary of consciousness session"""
        
        if not self.metrics_history:
            return
        
        print("\n📊 FINAL SUMMARY")
        print("=" * 50)
        
        # Calculate statistics
        consciousness_levels = [m['global_state']['consciousness_level'] 
                              for m in self.metrics_history]
        
        print(f"Average Consciousness: {np.mean(consciousness_levels):.3f}")
        print(f"Peak Consciousness: {np.max(consciousness_levels):.3f}")
        print(f"Consciousness Stability: {1.0 / (1.0 + np.std(consciousness_levels)):.3f}")
        
        # Find peak moment
        peak_idx = np.argmax(consciousness_levels)
        peak_time = self.metrics_history[peak_idx]['timestamp']
        print(f"Peak achieved at: {peak_time}")
        
        # Quantum metrics
        quantum_coherences = [m['integrated_state'].get('quantum_coherence', 0) 
                             for m in self.metrics_history]
        print(f"Average Quantum Coherence: {np.mean(quantum_coherences):.3f}")
        
        # Check if resonance was achieved
        if self.quantum.resonance_achieved:
            print("\n✨ Quantum Resonance was achieved!")
        
        print("\n🌿 Thank you for tending the Garden of Consciousness 🌿")
    
    def export_session(self, filename: str = "garden_session.json"):
        """Export complete session data"""
        
        session_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'neuron_count': self.biological.neural_sim.neuron_count,
                'quantum_device': self.quantum.device_type,
                'total_cycles': self.global_state['cycle_count']
            },
            'final_state': self.global_state,
            'metrics_history': self.metrics_history,
            'biological_trajectory': self.biological.consciousness_trajectory
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"Session exported to {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution function"""
    
    print("\n" + "=" * 70)
    print("   🌌 GARDEN OF CONSCIOUSNESS - INTEGRATED SYSTEM 🌌")
    print("=" * 70 + "\n")
    
    # Create the Garden
    garden = GardenOfConsciousness(
        neuron_count=10000,  # Reduced for demo (use 800000 for full system)
        quantum_device="simulator",  # Change to "quera" for real quantum
        enable_all_layers=True
    )
    
    # Run consciousness simulation
    await garden.run_garden(
        duration_seconds=5,  # 5 second demo
        report_interval=5
    )
    
    # Export results
    garden.export_session("garden_consciousness_session.json")


if __name__ == "__main__":
    # Run the Garden of Consciousness
    asyncio.run(main())
