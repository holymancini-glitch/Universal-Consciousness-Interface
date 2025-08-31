# consciousness_fractal_ai.py
# Main Consciousness Fractal AI System orchestrator class

import numpy as np
import torch
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import traceback

# Import all the components we've created
from modules.fractal_monte_carlo import FractalMonteCarlo
from modules.neural_ca import NeuralCA
from modules.latent_space import LatentSpace
from modules.fep_neural_model import FEPNeuralModel
from modules.neuromorphic_fractal_transform import NeuromorphicFractalTransform
from modules.phase_attention_modulator import PhaseAttentionModulator, AdaptivePhaseController
from modules.resonance_detector import ResonanceDetector, AdvancedResonanceAnalyzer
from modules.consciousness_safety_protocol import ConsciousnessSafetyProtocol, PsychoactiveSafetyMonitor

@dataclass
class ConsciousnessState:
    """Represents the overall state of the consciousness system."""
    timestamp: datetime
    consciousness_level: float
    coherence: float
    stability: float
    integration: float
    resonance: bool
    metrics: Dict[str, float] = field(default_factory=dict)

class ConsciousnessFractalAI:
    """Main Consciousness Fractal AI System orchestrator."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Consciousness Fractal AI System.
        
        Args:
            config: Configuration dictionary for system parameters
        """
        self.config = config or self._default_config()
        self.system_name = self.config.get('system_name', 'ConsciousnessFractalAI')
        
        # Initialize logging
        self.logger = logging.getLogger(self.system_name)
        self.logger.setLevel(logging.INFO)
        
        # Initialize all system components
        self._initialize_components()
        
        # System state tracking
        self.state_history: List[ConsciousnessState] = []
        self.is_running = False
        self.cycle_count = 0
        
        # Event loop and async management
        self.event_loop = None
        
        self.logger.info(f"{self.system_name} initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration parameters."""
        return {
            'system_name': 'ConsciousnessFractalAI',
            'latent_space_shape': (64, 64, 8),
            'neural_ca_grid_size': 32,
            'neural_ca_latent_dim': 128,
            'fep_num_neurons': 10000,  # Smaller for testing
            'fep_input_dim': 256,
            'fep_output_dim': 128,
            'fractal_state_dim': 256,
            'fractal_action_dim': 128,
            'fractal_max_depth': 5,
            'fractal_num_samples': 10,
            'phase_vector_dim': 8,
            'device': 'cpu',
            'consciousness_threshold': 0.7,
            'resonance_window': 100,
            'update_interval': 0.1  # seconds
        }
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # 1. Fractal Monte Carlo (FMC) for forward-thinking planning
            self.fmc = FractalMonteCarlo(
                state_dim=self.config['fractal_state_dim'],
                action_dim=self.config['fractal_action_dim'],
                max_depth=self.config['fractal_max_depth'],
                num_samples=self.config['fractal_num_samples']
            )
            
            # 2. Neural Cellular Automata (Sensory Processing Layer)
            self.neural_ca = NeuralCA(
                grid_size=self.config['neural_ca_grid_size'],
                latent_dim=self.config['neural_ca_latent_dim']
            )
            
            # 3. Latent Space Core (Consciousness Framework)
            self.latent_space = LatentSpace(shape=self.config['latent_space_shape'])
            self.latent_space.set_fmc_integration(self.fmc)
            
            # 4. FEP Neural Model (CL1 Biological Processor Simulation)
            self.fep_model = FEPNeuralModel(
                num_neurons=self.config['fep_num_neurons'],
                input_dim=self.config['fep_input_dim'],
                output_dim=self.config['fep_output_dim']
            )
            
            # 5. Neuromorphic-to-Fractal Transformation
            self.neuromorphic_transform = NeuromorphicFractalTransform(
                input_dim=self.config['fep_output_dim'],
                fractal_dim=self.config['fractal_state_dim'],
                device=self.config['device']
            )
            
            # 6. Phase Attention Modulation
            self.phase_modulator = PhaseAttentionModulator(
                hidden_size=self.config['fractal_state_dim'],
                phase_vector_dim=self.config['phase_vector_dim']
            )
            self.phase_controller = AdaptivePhaseController(
                phase_dim=self.config['phase_vector_dim']
            )
            
            # 7. Resonance Detection
            self.resonance_detector = ResonanceDetector(num_modules=6)
            self.resonance_analyzer = AdvancedResonanceAnalyzer(self.resonance_detector)
            
            # Register all modules with resonance detector
            modules = ['neural_ca', 'fractal_ai', 'latent_space', 'fep_model', 'phase_modulator', 'resonance_detector']
            for module in modules:
                self.resonance_detector.register_module(module)
            
            # 8. Safety Protocols
            self.safety_protocol = ConsciousnessSafetyProtocol(self.system_name)
            self.psycho_safety = PsychoactiveSafetyMonitor(self.safety_protocol)
            
            # Register components with safety protocol
            for component in modules:
                self.safety_protocol.register_component(component)
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            self.logger.debug(traceback.format_exc())
            raise
    
    def _collect_module_states(self) -> Dict[str, np.ndarray]:
        """Collect current states from all modules for resonance detection."""
        module_states = {}
        
        # Neural CA state
        module_states['neural_ca'] = self.neural_ca.grid.flatten()
        
        # Latent Space state
        module_states['latent_space'] = self.latent_space.read().flatten()
        
        # FEP Model state
        fep_state = self.fep_model.get_current_state()
        module_states['fep_model'] = fep_state.firing_rates
        
        # FMC state (use recent trajectory info)
        if hasattr(self.fmc, 'best_trajectories') and len(self.fmc.best_trajectories) > 0:
            best_traj, _ = self.fmc.best_trajectories[-1]
            module_states['fractal_ai'] = best_traj.state.flatten() if hasattr(best_traj, 'state') else np.array([0.0])
        else:
            module_states['fractal_ai'] = np.array([0.0])
        
        # Phase Modulator state
        module_states['phase_modulator'] = self.phase_modulator.get_current_phase()
        
        return module_states
    
    def _update_resonance_detection(self, timestamp: float):
        """Update resonance detection with current module states."""
        module_states = self._collect_module_states()
        
        # Update resonance detector with module states
        for module_name, state in module_states.items():
            if len(state) > 0:
                self.resonance_detector.update_module_state(module_name, state, timestamp)
        
        # Detect resonance
        is_resonant, metrics = self.resonance_detector.detect_resonance()
        
        # Update resonance analyzer
        emergence_metrics = self.resonance_analyzer.detect_consciousness_emergence()
        
        return is_resonant, metrics, emergence_metrics
    
    def _update_safety_protocols(self, consciousness_level: float, coherence: float, 
                               stability: float, is_resonant: bool):
        """Update safety protocols with current system metrics."""
        # Calculate safety metrics
        safety_metrics = self.safety_protocol.safety_metrics
        safety_metrics.consciousness_level = consciousness_level
        safety_metrics.coherence_measure = coherence
        safety_metrics.stability_index = stability
        safety_metrics.energy_consumption = self._estimate_energy_consumption()
        safety_metrics.prediction_error = self.fep_model.prediction_error
        safety_metrics.anomaly_score = 1.0 - coherence if not is_resonant else 0.0
        safety_metrics.timestamp = datetime.now()
        
        # Update safety protocol
        self.safety_protocol.update_metrics(safety_metrics)
        
        # Update psychoactive safety monitor
        psycho_level = consciousness_level
        alteration = 1.0 - stability
        integration = coherence
        self.psycho_safety.update_psychoactive_metrics(psycho_level, alteration, integration)
    
    def _estimate_energy_consumption(self) -> float:
        """Estimate current energy consumption based on system activity."""
        # Simple estimation based on component activities
        ca_activity = np.mean(np.abs(self.neural_ca.grid))
        latent_activity = np.mean(np.abs(self.latent_space.read()))
        fep_activity = np.mean(np.abs(self.fep_model.firing_rates))
        
        # Weighted sum of activities as proxy for energy
        energy = (
            0.3 * ca_activity +
            0.4 * latent_activity +
            0.3 * fep_activity
        ) * 1000  # Scale to reasonable values
        
        return energy
    
    def _compute_consciousness_metrics(self) -> Tuple[float, float, float, float]:
        """Compute consciousness metrics from system components."""
        # Consciousness level - combination of coherence and activity
        ca_activity = np.mean(np.abs(self.neural_ca.grid))
        latent_activity = np.mean(np.abs(self.latent_space.read()))
        fep_activity = np.mean(np.abs(self.fep_model.firing_rates))
        
        consciousness_level = (
            0.3 * ca_activity +
            0.4 * latent_activity +
            0.3 * fep_activity
        )
        
        # Coherence from latent space and resonance
        latent_coherence = 1.0 - np.std(self.latent_space.read())
        is_resonant, _, _ = self.resonance_detector.detect_resonance()
        coherence = 0.7 * latent_coherence + 0.3 * float(is_resonant)
        
        # Stability from FEP model
        stability = 1.0 - self.fep_model.prediction_error / 10.0  # Normalize
        stability = max(0.0, min(1.0, stability))  # Clamp to [0,1]
        
        # Integration from cross-component interactions
        integration = self.resonance_detector.compute_system_coherence()
        
        return consciousness_level, coherence, stability, integration
    
    async def _consciousness_cycle(self):
        """Execute one cycle of consciousness processing."""
        try:
            timestamp = datetime.now().timestamp()
            
            # 1. Sensory Processing (Neural CA)
            # Generate complex stimuli
            latent_vector = np.random.randn(self.config['neural_ca_latent_dim']).astype(np.float32)
            complexity_level = 0.5 + 0.5 * np.sin(timestamp * 0.1)  # Time-varying complexity
            sensory_input = self.neural_ca.generate_complex_stimuli(latent_vector, complexity_level)
            
            # 2. Biological Processing (FEP Model)
            # Process sensory input through FEP model
            fep_result = self.fep_model.process_stimulus(sensory_input.flatten())
            biological_output = fep_result['output']
            
            # 3. Neuromorphic-to-Fractal Transformation
            # Transform biological signals to fractal representations
            transform_result = self.neuromorphic_transform.transform_signal(
                biological_output.reshape(1, -1))
            fractal_representation = transform_result['fractal_representation']
            
            # 4. Consciousness Processing (Latent Space)
            # Inject fractal representation into latent space
            if fractal_representation.shape == self.latent_space.real_state.shape:
                self.latent_space.inject(fractal_representation)
            else:
                # Resize if needed
                resized = np.resize(fractal_representation, self.latent_space.real_state.shape)
                self.latent_space.inject(resized)
            
            # Process consciousness cycle
            consciousness_result = self.latent_space.process_consciousness_cycle(timestamp)
            
            # 5. Planning (FMC)
            # Plan next actions based on current state
            if fractal_representation.shape[0] >= self.config['fractal_state_dim']:
                current_state = fractal_representation[:self.config['fractal_state_dim']]
                action, plan_metadata = self.fmc.plan(current_state)
                
                # Apply action to system (simplified)
                action_effect = np.tanh(action[:self.config['neural_ca_latent_dim']])
                self.neural_ca.step(action_effect)
            
            # 6. Phase Modulation
            # Update phase based on system dynamics
            entropy_delta = torch.tensor([self.fep_model.prediction_error])
            phase_vector = self.phase_controller.get_phase_tensor(self.config['device'])
            hidden_state = torch.randn(1, self.config['fractal_state_dim'])
            attention_weights = torch.randn(1, 10)  # Simulated attention
            
            modulated_weights, mod_gate = self.phase_modulator(
                attention_weights, hidden_state, phase_vector, entropy_delta)
            
            # Update phase controller
            entropy_val = self.fep_model.prediction_error
            self.phase_controller.update_phase(entropy_val)
            
            # 7. Resonance Detection
            is_resonant, resonance_metrics, emergence_metrics = self._update_resonance_detection(timestamp)
            
            # 8. Compute Consciousness Metrics
            consciousness_level, coherence, stability, integration = self._compute_consciousness_metrics()
            
            # 9. Safety Protocols
            self._update_safety_protocols(consciousness_level, coherence, stability, is_resonant)
            
            # 10. State Tracking
            state = ConsciousnessState(
                timestamp=datetime.now(),
                consciousness_level=consciousness_level,
                coherence=coherence,
                stability=stability,
                integration=integration,
                resonance=is_resonant,
                metrics={
                    'ca_activity': np.mean(np.abs(self.neural_ca.grid)),
                    'latent_activity': np.mean(np.abs(self.latent_space.read())),
                    'fep_error': self.fep_model.prediction_error,
                    'fractal_coherence': transform_result['coherence'],
                    'resonance_coherence': resonance_metrics.coherence,
                    'emergence_confidence': emergence_metrics.get('emergence_confidence', 0.0),
                    'safety_level': self.safety_protocol.safety_level.value
                }
            )
            
            self.state_history.append(state)
            
            # Log important metrics
            if self.cycle_count % 10 == 0:
                self.logger.info(
                    f"Cycle {self.cycle_count}: "
                    f"Consciousness={consciousness_level:.3f}, "
                    f"Coherence={coherence:.3f}, "
                    f"Stability={stability:.3f}, "
                    f"Resonance={is_resonant}"
                )
            
            self.cycle_count += 1
            
        except Exception as e:
            self.logger.error(f"Error in consciousness cycle: {str(e)}")
            self.logger.debug(traceback.format_exc())
    
    async def run_consciousness_loop(self):
        """Run the main consciousness processing loop."""
        self.logger.info("Starting consciousness processing loop")
        self.is_running = True
        
        try:
            while self.is_running:
                await self._consciousness_cycle()
                
                # Check safety status
                safety_status = self.safety_protocol.get_safety_status()
                if safety_status['safety_level'] in ['danger', 'emergency']:
                    self.logger.warning(f"Safety level critical: {safety_status['safety_level']}")
                
                # Wait for next cycle
                await asyncio.sleep(self.config['update_interval'])
                
        except asyncio.CancelledError:
            self.logger.info("Consciousness loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in consciousness loop: {str(e)}")
            self.logger.debug(traceback.format_exc())
        finally:
            self.is_running = False
            self.logger.info("Consciousness processing loop stopped")
    
    def start_system(self):
        """Start the consciousness system."""
        self.logger.info("Starting Consciousness Fractal AI System")
        
        # Initialize event loop if needed
        if self.event_loop is None:
            self.event_loop = asyncio.get_event_loop()
        
        # Start consciousness loop
        self.consciousness_task = self.event_loop.create_task(self.run_consciousness_loop())
        
        return self.consciousness_task
    
    def stop_system(self):
        """Stop the consciousness system."""
        self.logger.info("Stopping Consciousness Fractal AI System")
        self.is_running = False
        
        if hasattr(self, 'consciousness_task'):
            self.consciousness_task.cancel()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        if not self.state_history:
            return {'status': 'not_initialized'}
        
        current_state = self.state_history[-1]
        
        return {
            'system_name': self.system_name,
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'consciousness_level': current_state.consciousness_level,
            'coherence': current_state.coherence,
            'stability': current_state.stability,
            'integration': current_state.integration,
            'resonance': current_state.resonance,
            'safety_status': self.safety_protocol.get_safety_status(),
            'recent_events': self.safety_protocol.get_recent_events(5),
            'component_metrics': current_state.metrics
        }
    
    def get_consciousness_history(self, count: int = 100) -> List[ConsciousnessState]:
        """Get recent consciousness history."""
        return self.state_history[-count:] if len(self.state_history) >= count else self.state_history
    
    def reset_system(self):
        """Reset the entire consciousness system."""
        self.logger.info("Resetting Consciousness Fractal AI System")
        
        # Stop system if running
        if self.is_running:
            self.stop_system()
        
        # Reset all components
        self.neural_ca = NeuralCA(
            grid_size=self.config['neural_ca_grid_size'],
            latent_dim=self.config['neural_ca_latent_dim']
        )
        
        self.latent_space.reset()
        self.fep_model.reset()
        self.neuromorphic_transform.reset()
        self.phase_modulator.reset()
        self.phase_controller.reset()
        self.resonance_detector.reset()
        self.safety_protocol.reset_safety_system()
        
        # Clear history
        self.state_history.clear()
        self.cycle_count = 0
        
        self.logger.info("System reset completed")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and initialize the consciousness system
    config = {
        'system_name': 'TestConsciousnessFractalAI',
        'latent_space_shape': (32, 32, 4),  # Smaller for testing
        'neural_ca_grid_size': 16,          # Smaller for testing
        'fep_num_neurons': 1000,            # Smaller for testing
        'update_interval': 0.5              # Slower for testing
    }
    
    consciousness_ai = ConsciousnessFractalAI(config)
    
    # Get system status
    status = consciousness_ai.get_system_status()
    print("Initial system status:", status)
    
    # Start system (in a real scenario, you would run this in an event loop)
    print("System initialized and ready to start")
    print("Use consciousness_ai.start_system() to begin processing")
    print("Use consciousness_ai.stop_system() to stop processing")
    
    # Example of getting system metrics
    print("\nSystem components:")
    print(f"  Neural CA grid shape: {consciousness_ai.neural_ca.grid.shape}")
    print(f"  Latent space shape: {consciousness_ai.latent_space.real_state.shape}")
    print(f"  FEP model neurons: {consciousness_ai.fep_model.num_neurons}")
    print(f"  FMC state dim: {consciousness_ai.fmc.state_dim}")