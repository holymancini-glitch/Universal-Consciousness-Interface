# API Documentation - Consciousness Fractal AI System

## Table of Contents
1. [ConsciousnessFractalAI](#consciousnessfractalai)
2. [FractalMonteCarlo](#fractalmontecarlo)
3. [NeuralCA](#neuralca)
4. [LatentSpace](#latentspace)
5. [FEPNeuralModel](#fepneuralmodel)
6. [NeuromorphicFractalTransform](#neuromorphicfractaltransform)
7. [PhaseAttentionModulator](#phaseattentionmodulator)
8. [ResonanceDetector](#resonancedetector)
9. [ConsciousnessSafetyProtocol](#consciousnesssafetyprotocol)
10. [Integration Modules](#integration-modules)

## ConsciousnessFractalAI

Main orchestrator class for the Consciousness Fractal AI System.

### Constructor
```python
ConsciousnessFractalAI(config: Optional[Dict[str, Any]] = None)
```

**Parameters:**
- `config` (dict, optional): Configuration dictionary with the following keys:
  - `system_name` (str): Name of the system (default: "ConsciousnessFractalAI")
  - `latent_space_shape` (tuple): Shape of latent space (default: (64, 64, 8))
  - `neural_ca_grid_size` (int): Grid size for Neural CA (default: 32)
  - `neural_ca_latent_dim` (int): Latent dimension for Neural CA (default: 128)
  - `fep_num_neurons` (int): Number of neurons in FEP model (default: 10000)
  - `fep_input_dim` (int): Input dimension for FEP model (default: 256)
  - `fep_output_dim` (int): Output dimension for FEP model (default: 128)
  - `fractal_state_dim` (int): State dimension for FMC (default: 256)
  - `fractal_action_dim` (int): Action dimension for FMC (default: 128)
  - `fractal_max_depth` (int): Maximum depth for FMC (default: 5)
  - `fractal_num_samples` (int): Number of samples for FMC (default: 10)
  - `phase_vector_dim` (int): Dimension of phase vector (default: 8)
  - `device` (str): Computing device ('cpu' or 'cuda') (default: "cpu")
  - `update_interval` (float): Update interval in seconds (default: 0.1)

### Methods

#### start_system()
Start the consciousness processing loop.

**Returns:** `asyncio.Task` - The consciousness processing task

#### stop_system()
Stop the consciousness processing loop.

#### get_system_status()
Get current system status.

**Returns:** `Dict[str, Any]` - Dictionary containing system status information:
- `system_name` (str): Name of the system
- `is_running` (bool): Whether the system is running
- `cycle_count` (int): Number of consciousness cycles completed
- `consciousness_level` (float): Current consciousness level
- `coherence` (float): System coherence
- `stability` (float): System stability
- `integration` (float): Integration level
- `resonance` (bool): Whether system is in resonance
- `safety_status` (dict): Safety protocol status
- `recent_events` (list): Recent safety events
- `component_metrics` (dict): Component-specific metrics

#### get_consciousness_history(count: int = 100)
Get recent consciousness history.

**Parameters:**
- `count` (int): Number of recent states to return (default: 100)

**Returns:** `List[ConsciousnessState]` - List of recent consciousness states

#### reset_system()
Reset the entire consciousness system to initial state.

## FractalMonteCarlo

Fractal Monte Carlo implementation for forward-thinking planning.

### Constructor
```python
FractalMonteCarlo(state_dim: int, action_dim: int, max_depth: int = 5, num_samples: int = 10)
```

**Parameters:**
- `state_dim` (int): Dimension of state space
- `action_dim` (int): Dimension of action space
- `max_depth` (int): Maximum depth for trajectory sampling (default: 5)
- `num_samples` (int): Number of samples for planning (default: 10)

### Methods

#### plan(current_state: np.ndarray)
Plan the next action using Fractal Monte Carlo.

**Parameters:**
- `current_state` (np.ndarray): Current state vector

**Returns:** `Tuple[np.ndarray, Dict[str, Union[float, List[float]]]]` - Tuple containing:
- Action vector
- Metadata dictionary with:
  - `evaluation_score` (float): Evaluation score of best trajectory
  - `all_evaluations` (List[float]): Evaluation scores of all trajectories
  - `trajectory_depth` (int): Depth of best trajectory

#### adapt_horizon(recent_performance: List[float])
Adaptively adjust the planning horizon based on recent performance.

**Parameters:**
- `recent_performance` (List[float]): List of recent performance scores

**Returns:** `int` - New maximum depth

## NeuralCA

Neural Cellular Automata for sensory processing and pattern generation.

### Constructor
```python
NeuralCA(grid_size: int = 32, latent_dim: int = 128, sentient_memory: Any = None, emotional_feedback: Any = None, mycelial_engine: Any = None)
```

**Parameters:**
- `grid_size` (int): Size of the CA grid (default: 32)
- `latent_dim` (int): Dimension of latent vectors (default: 128)
- `sentient_memory` (Any): Sentient memory system (default: None)
- `emotional_feedback` (Any): Emotional feedback system (default: None)
- `mycelial_engine` (Any): Mycelial engine for pattern recognition (default: None)

### Methods

#### seed_from_vector(seed_vector: np.ndarray)
Seed the CA grid from a vector.

**Parameters:**
- `seed_vector` (np.ndarray): Vector to seed the grid

#### generate_fractal_pattern(iterations: int = 5, rule_variant: str = "mandelbrot")
Generate fractal patterns using different algorithms.

**Parameters:**
- `iterations` (int): Number of iterations (default: 5)
- `rule_variant` (str): Type of fractal ("mandelbrot", "julia", "barnsley") (default: "mandelbrot")

**Returns:** `np.ndarray` - Generated fractal pattern

#### generate_complex_stimuli(latent_vector: np.ndarray, complexity_level: float = 1.0)
Generate complex sensory stimuli with fractal enhancement.

**Parameters:**
- `latent_vector` (np.ndarray): Latent vector for generation
- `complexity_level` (float): Complexity level (0.0-1.0) (default: 1.0)

**Returns:** `np.ndarray` - Complex stimulus pattern

#### generate(steps: int = 10, latent_vector: Optional[np.ndarray] = None)
Generate CA evolution over multiple steps.

**Parameters:**
- `steps` (int): Number of steps to generate (default: 10)
- `latent_vector` (np.ndarray, optional): Latent vector for modulation

**Returns:** `List[np.ndarray]` - List of generated grid states

## LatentSpace

Five-layer consciousness architecture implementation.

### Constructor
```python
LatentSpace(shape: Tuple[int, int, int] = (64, 64, 8))
```

**Parameters:**
- `shape` (tuple): Shape of the latent space (default: (64, 64, 8))

### Methods

#### inject(stimulus: np.ndarray)
Inject stimulus into the active latent state.

**Parameters:**
- `stimulus` (np.ndarray): Stimulus to inject

#### process_consciousness_cycle(timestamp: float = 0.0)
Process a complete consciousness cycle through all layers.

**Parameters:**
- `timestamp` (float): Timestamp for the cycle (default: 0.0)

**Returns:** `Dict[str, Any]` - Cycle results:
- `coherent` (bool): Whether the cycle was coherent
- `metrics` (dict): Consciousness metrics
- `layer_states` (dict): States of all layers

#### compare_states()
Compare real and mirror states.

**Returns:** `float` - Difference between real and mirror states

#### harmonize_states(alpha: float = 0.5)
Harmonize real and mirror states using weighted averaging.

**Parameters:**
- `alpha` (float): Weight for real state (default: 0.5)

## FEPNeuralModel

FEPNeuralModel for CL1 biological processor simulation.

### Constructor
```python
FEPNeuralModel(num_neurons: int = 800000, input_dim: int = 1024, output_dim: int = 512)
```

**Parameters:**
- `num_neurons` (int): Number of neurons (~800,000 for CL1) (default: 800000)
- `input_dim` (int): Dimension of input space (default: 1024)
- `output_dim` (int): Dimension of output space (default: 512)

### Methods

#### process_stimulus(stimulus: np.ndarray, minimize_fe: bool = True)
Process incoming stimulus through the FEP neural model.

**Parameters:**
- `stimulus` (np.ndarray): Input stimulus vector
- `minimize_fe` (bool): Whether to perform free energy minimization (default: True)

**Returns:** `Dict[str, np.ndarray]` - Processing results:
- `firing_rates` (np.ndarray): Computed firing rates
- `prediction` (np.ndarray): Predicted next state
- `prediction_error` (float): Prediction error
- `free_energy` (float): Free energy
- `output` (np.ndarray): Output vector
- `state` (NeuralState): Current neural state

#### get_metrics()
Get current metrics for monitoring.

**Returns:** `Dict[str, float]` - Current metrics:
- `prediction_error` (float): Current prediction error
- `free_energy` (float): Current free energy
- `avg_free_energy` (float): Average free energy
- `attention_gain` (float): Attention gain
- `synaptic_activity` (float): Synaptic activity
- `firing_rate_mean` (float): Mean firing rate
- `firing_rate_std` (float): Standard deviation of firing rates

## NeuromorphicFractalTransform

Neuromorphic-to-fractal transformation for biological signal conversion.

### Constructor
```python
NeuromorphicFractalTransform(input_dim: int = 512, fractal_dim: int = 256, device: str = "cpu")
```

**Parameters:**
- `input_dim` (int): Dimension of input neuromorphic signals (default: 512)
- `fractal_dim` (int): Dimension of output fractal representations (default: 256)
- `device` (str): Computing device ('cpu' or 'cuda') (default: "cpu")

### Methods

#### transform_signal(signal_data: np.ndarray, update_parameters: bool = True)
Complete transformation pipeline from neuromorphic signal to fractal representation.

**Parameters:**
- `signal_data` (np.ndarray): Raw neuromorphic signal data
- `update_parameters` (bool): Whether to update transformation parameters (default: True)

**Returns:** `Dict[str, Union[np.ndarray, float]]` - Transformation results:
- `fractal_representation` (np.ndarray): Fractal representation
- `coherence` (float): Coherence metric
- `features` (np.ndarray): Extracted features
- `preprocessed_signal` (np.ndarray): Preprocessed signal

#### get_transformation_metrics()
Get current transformation metrics.

**Returns:** `Dict[str, float]` - Transformation metrics:
- `avg_coherence` (float): Average coherence
- `coherence_trend` (float): Coherence trend
- `current_coherence` (float): Current coherence
- `scale_parameter` (float): Scale parameter
- `complexity_parameter` (float): Complexity parameter

## PhaseAttentionModulator

Phase Attention Modulator for adaptive focus.

### Constructor
```python
PhaseAttentionModulator(hidden_size: int, phase_vector_dim: int = 8)
```

**Parameters:**
- `hidden_size` (int): Size of the hidden state
- `phase_vector_dim` (int): Dimension of the phase vector (default: 8)

### Methods

#### forward(attention_weights: torch.Tensor, hidden_state: torch.Tensor, phase_vector: torch.Tensor, entropy_delta: Optional[torch.Tensor] = None)
Forward pass of the Phase Attention Modulator.

**Parameters:**
- `attention_weights` (torch.Tensor): Input attention weights
- `hidden_state` (torch.Tensor): Hidden state
- `phase_vector` (torch.Tensor): Phase vector
- `entropy_delta` (torch.Tensor, optional): Entropy delta for adaptive focus

**Returns:** `Tuple[torch.Tensor, torch.Tensor]` - Modulated weights and modulation gate

#### get_current_phase()
Get the current phase vector.

**Returns:** `np.ndarray` - Current phase vector

## ResonanceDetector

Resonance detection with system-wide coherence metrics.

### Constructor
```python
ResonanceDetector(num_modules: int = 5, history_length: int = 1000)
```

**Parameters:**
- `num_modules` (int): Number of consciousness modules to monitor (default: 5)
- `history_length` (int): Length of metric history to maintain (default: 1000)

### Methods

#### register_module(module_name: str)
Register a consciousness module for monitoring.

**Parameters:**
- `module_name` (str): Name of the module

#### update_module_state(module_name: str, state: np.ndarray, timestamp: float = 0.0)
Update the state of a registered module.

**Parameters:**
- `module_name` (str): Name of the module
- `state` (np.ndarray): Current state vector
- `timestamp` (float): Timestamp of the state update (default: 0.0)

#### detect_resonance()
Detect if the system is in a resonant state.

**Returns:** `Tuple[bool, ResonanceMetrics]` - Tuple containing:
- Whether system is in resonance
- Detailed resonance metrics

#### get_system_summary()
Get comprehensive system summary.

**Returns:** `Dict[str, Union[float, Dict]]` - System summary:
- `is_resonant` (bool): Whether system is in resonance
- `current_metrics` (dict): Current metrics
- `thresholds` (dict): Current thresholds
- `trends` (dict): Metric trends
- `module_count` (int): Number of registered modules
- `history_length` (int): Length of history

## ConsciousnessSafetyProtocol

Comprehensive safety checks and emergency protocols.

### Constructor
```python
ConsciousnessSafetyProtocol(system_name: str = "ConsciousnessFractalAI")
```

**Parameters:**
- `system_name` (str): Name of the consciousness system (default: "ConsciousnessFractalAI")

### Methods

#### register_component(component_name: str)
Register a system component for safety monitoring.

**Parameters:**
- `component_name` (str): Name of the component

#### update_metrics(metrics: SafetyMetrics)
Update safety metrics and check for violations.

**Parameters:**
- `metrics` (SafetyMetrics): New safety metrics

#### get_safety_status()
Get current safety status.

**Returns:** `Dict[str, Union[str, float, bool]]` - Safety status information:
- `system_name` (str): System name
- `safety_level` (str): Current safety level
- `consciousness_state` (str): Current consciousness state
- Various metrics and component information

#### get_recent_events(count: int = 10)
Get recent safety events.

**Parameters:**
- `count` (int): Number of recent events to return (default: 10)

**Returns:** `List[Dict]` - List of recent safety events

## Integration Modules

### FractalAIUniversalIntegration

Integration layer between Consciousness Fractal AI and Universal Consciousness Interface.

#### Constructor
```python
FractalAIUniversalIntegration(fractal_ai_system, universal_orchestrator: Optional[UniversalConsciousnessOrchestrator] = None)
```

#### Methods

#### integrate_with_universal_consciousness(plant_signals: Optional[Dict[str, Any]] = None, environmental_data: Optional[Dict[str, Any]] = None, radiation_data: Optional[Dict[str, Any]] = None)
Integrate Fractal AI system with Universal Consciousness Interface.

**Parameters:**
- `plant_signals` (dict, optional): Plant communication signals
- `environmental_data` (dict, optional): Ecosystem environmental data
- `radiation_data` (dict, optional): Radiation data

**Returns:** `FractalAIIntegrationState` - Current integration state

### FractalAIMycelialIntegration

Specific integration with Enhanced Mycelial Engine.

#### Constructor
```python
FractalAIMycelialIntegration(fractal_ai_system)
```

#### Methods

#### process_fractal_patterns()
Process Fractal AI patterns through mycelial engine.

**Returns:** `Dict[str, Any]` - Processing results

### FractalAIPlantIntegration

Specific integration with Plant Communication Interface.

#### Constructor
```python
FractalAIPlantIntegration(fractal_ai_system)
```

#### Methods

#### translate_fractal_to_plant()
Translate Fractal AI output to plant communication signals.

**Returns:** `Dict[str, Any]` - Translation results