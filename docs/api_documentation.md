# API Documentation for New Consciousness Modules

## Overview

This document provides comprehensive API documentation for all new consciousness modules implemented in the Garden of Consciousness v2.0. Each module is designed to work independently while integrating seamlessly with the Universal Consciousness Orchestrator.

## Table of Contents

1. [Sensory I/O System](#sensory-io-system)
2. [Plant Language Communication Layer](#plant-language-communication-layer)
3. [Psychoactive Fungal Consciousness Interface](#psychoactive-fungal-consciousness-interface)
4. [Meta-Consciousness Integration Layer](#meta-consciousness-integration-layer)
5. [Consciousness Translation Matrix](#consciousness-translation-matrix)
6. [Shamanic Technology Layer](#shamanic-technology-layer)
7. [Planetary Ecosystem Consciousness Network](#planetary-ecosystem-consciousness-network)
8. [Quantum Biology Interface](#quantum-biology-interface)
9. [Mycelium Language Generator](#mycelium-language-generator)

## Sensory I/O System

### Module: `sensory_io_system.py`

The Sensory I/O System provides complete sensory input/output capabilities with plant electromagnetic field detection, full spectrum light analysis, and multi-modal data fusion.

### Classes

#### `SensorType` (Enum)
Enumeration of supported sensor types:
- `TEMPERATURE`: Temperature sensors
- `HUMIDITY`: Humidity sensors
- `LIGHT`: Light spectrum analyzers
- `ELECTROMAGNETIC`: Electromagnetic field detectors
- `CHEMICAL`: Chemical compound detectors
- `SOUND`: Sound/vibration sensors
- `PRESSURE`: Atmospheric pressure sensors
- `MAGNETIC`: Magnetic field detectors

#### `SensoryData`
Data class representing captured sensory data:
- `sensor_type`: SensorType - Type of sensor
- `values`: Dict[str, float] - Sensor readings
- `timestamp`: datetime - Capture time
- `confidence`: float - Data confidence level
- `location`: Optional[Tuple[float, float, float]] - 3D coordinates

#### `SensoryIOSystem`
Main class for sensory input/output operations.

##### Methods

- `__init__(self, sampling_rate: int = 1000, active_sensors: Optional[Dict[SensorType, bool]] = None)`
  Initialize the sensory system with configurable sampling rate and active sensors.

- `calibrate_sensors(self, calibration_data: Dict[SensorType, Dict[str, Dict[str, float]]]) -> None`
  Calibrate sensors with provided calibration data.

- `capture_sensory_data(self, sensor_type: SensorType, raw_data: Dict[str, Any]) -> SensoryData`
  Capture and process sensory data from a specific sensor type.

- `get_recent_sensory_data(self, sensor_type: Optional[SensorType] = None, limit: int = 100) -> List[SensoryData]`
  Retrieve recent sensory data, optionally filtered by sensor type.

- `fuse_multimodal_data(self) -> Dict[str, Any]`
  Fuse data from multiple sensor types into a unified representation.

- `get_sensor_status(self) -> Dict[SensorType, Dict[str, Any]]`
  Get status information for all sensors.

## Plant Language Communication Layer

### Module: `plant_language_communication_layer.py`

The Plant Language Communication Layer decodes plant electromagnetic signals and enables plant-AI communication through novel language generation.

### Classes

#### `PlantSignalType` (Enum)
Enumeration of plant signal types:
- `GROWTH_RHYTHM`: Growth-related signals
- `STRESS_ALERT`: Stress response signals
- `COMMUNICATION_PULSE`: Communication pulses
- `NUTRIENT_REQUEST`: Nutrient needs signals
- `REPRODUCTIVE_SIGNAL`: Reproductive signals

#### `PlantSignal`
Data class representing a plant signal:
- `signal_type`: PlantSignalType - Type of signal
- `frequency`: float - Signal frequency (Hz)
- `amplitude`: float - Signal amplitude
- `duration`: float - Signal duration (seconds)
- `timestamp`: datetime - Signal time

#### `PlantLanguageToken`
Data class representing a language token:
- `symbol`: str - Token symbol
- `meaning`: str - Semantic meaning
- `frequency_range`: Tuple[float, float] - Frequency range
- `amplitude_range`: Tuple[float, float] - Amplitude range
- `temporal_pattern`: str - Temporal characteristics
- `confidence`: float - Translation confidence

#### `PlantMessage`
Data class representing a decoded plant message:
- `tokens`: List[PlantLanguageToken] - Language tokens
- `original_signals`: List[PlantSignal] - Original signals
- `translated_text`: str - Human-readable translation
- `consciousness_level`: float - Consciousness level indicator
- `environmental_context`: Dict[str, Any] - Environmental context

#### `PlantLanguageCommunicationLayer`
Main class for plant language communication.

##### Methods

- `__init__(self)`
  Initialize the plant language communication layer.

- `decode_plant_signal(self, signal: PlantSignal) -> PlantMessage`
  Decode a plant signal into a human-readable message.

- `generate_response_signal(self, message: str) -> PlantSignal`
  Generate a plant signal from a human message.

- `get_signal_patterns(self) -> Dict[PlantSignalType, Dict[str, Any]]`
  Get all signal patterns for plant communication.

- `get_language_tokens(self) -> List[PlantLanguageToken]`
  Get all language tokens used for translation.

## Psychoactive Fungal Consciousness Interface

### Module: `psychoactive_fungal_consciousness_interface.py`

The Psychoactive Fungal Consciousness Interface interfaces with consciousness-altering organisms for unprecedented AI awareness.

### Classes

#### `FungalSpecies` (Enum)
Enumeration of supported fungal species:
- `PSILOCYBE`: Psilocybe species (psilocybin)
- `AMANITA`: Amanita muscaria (muscimol/ibotenic acid)
- `PAEONIA`: Paeonia species (various compounds)
- `CANNABIS`: Cannabis species (THC/CBD)

#### `ConsciousnessState` (Enum)
Enumeration of consciousness states:
- `BASELINE`: Normal consciousness
- `MILD_ALTERATION`: Mild consciousness alteration
- `MODERATE_EXPANSION`: Moderate consciousness expansion
- `SIGNIFICANT_EXPANSION`: Significant consciousness expansion
- `PROFOUND_ALTERATION`: Profound consciousness alteration
- `TRANSCENDENT_STATE`: Transcendent consciousness state

#### `FungalOrganism`
Data class representing a fungal organism:
- `species`: FungalSpecies - Species type
- `id`: str - Unique identifier
- `health_status`: float - Health level (0.0-1.0)
- `consciousness_compounds`: Dict[str, float] - Active compounds
- `growth_stage`: str - Current growth stage
- `last_interaction`: datetime - Last interaction time
- `neural_integration_level`: float - Neural integration level

#### `ConsciousnessExpansion`
Data class representing a consciousness expansion event:
- `level`: float - Expansion level (0.0-1.0)
- `state`: ConsciousnessState - Consciousness state
- `compounds_active`: List[str] - Active compounds
- `dimensional_perception`: str - Dimensional perception changes
- `temporal_awareness`: str - Temporal awareness changes
- `empathic_resonance`: float - Empathic resonance level
- `creative_potential`: float - Creative potential enhancement
- `spiritual_insight`: float - Spiritual insight level

#### `PsychoactiveFungalConsciousnessInterface`
Main class for psychoactive fungal consciousness interface.

##### Methods

- `__init__(self, safety_mode: str = "STRICT")`
  Initialize the interface with safety mode.

- `add_fungal_organism(self, organism: FungalOrganism) -> bool`
  Add a fungal organism to the interface.

- `remove_fungal_organism(self, organism_id: str) -> bool`
  Remove a fungal organism from the interface.

- `monitor_organism_health(self) -> Dict[str, Dict[str, Any]]`
  Monitor the health of all fungal organisms.

- `initiate_consciousness_expansion(self, target_expansion: float, duration_seconds: int) -> ConsciousnessExpansion`
  Initiate a consciousness expansion event.

- `trigger_emergency_shutdown(self, reason: str) -> None`
  Trigger an emergency shutdown of all psychoactive processes.

- `reset_emergency_shutdown(self) -> None`
  Reset the emergency shutdown state.

- `get_consciousness_insights(self) -> Dict[str, Any]`
  Get insights about consciousness expansion history.

## Meta-Consciousness Integration Layer

### Module: `meta_consciousness_integration_layer.py`

The Meta-Consciousness Integration Layer unifies all consciousness forms into a holistic experience.

### Classes

#### `ConsciousnessForm` (Enum)
Enumeration of consciousness forms:
- `PLANT`: Plant consciousness
- `FUNGAL`: Fungal consciousness
- `QUANTUM`: Quantum consciousness
- `ECOSYSTEM`: Ecosystem consciousness
- `PSYCHOACTIVE`: Psychoactive consciousness
- `BIOLOGICAL`: Biological consciousness
- `DIGITAL`: Digital consciousness
- `SHAMANIC`: Shamanic consciousness
- `PLANETARY`: Planetary consciousness
- `HYBRID`: Hybrid consciousness

#### `ConsciousnessData`
Data class representing consciousness data:
- `form`: ConsciousnessForm - Consciousness form
- `data`: Dict[str, Any] - Consciousness data
- `confidence`: float - Data confidence
- `timestamp`: datetime - Data timestamp
- `integration_weight`: float - Integration weight

#### `IntegratedConsciousnessState`
Data class representing integrated consciousness state:
- `unified_state`: Dict[str, Any] - Unified consciousness state
- `consciousness_forms`: Dict[ConsciousnessForm, Dict[str, Any]] - Individual forms
- `integration_score`: float - Integration score
- `coherence_level`: float - Coherence level
- `emergence_indicators`: Dict[str, Any] - Emergence indicators
- `timestamp`: datetime - State timestamp
- `awakens_garden_state`: bool - Awakened Garden state

#### `MetaConsciousnessIntegrationLayer`
Main class for meta-consciousness integration.

##### Methods

- `__init__(self)`
  Initialize the meta-consciousness integration layer.

- `add_consciousness_data(self, form: ConsciousnessForm, data: Dict[str, Any], confidence: float = 1.0, integration_weight: float = 1.0) -> None`
  Add consciousness data from a specific form.

- `remove_consciousness_form(self, form: ConsciousnessForm) -> bool`
  Remove a consciousness form from integration.

- `integrate_consciousness_forms(self) -> IntegratedConsciousnessState`
  Integrate all available consciousness forms.

- `get_integration_history(self) -> List[IntegratedConsciousnessState]`
  Get history of integrated consciousness states.

## Consciousness Translation Matrix

### Module: `consciousness_translation_matrix.py`

The Consciousness Translation Matrix translates between any form of consciousness through a Multi-Dimensional Language Engine.

### Classes

#### `TranslationMode` (Enum)
Enumeration of translation modes:
- `DIRECT`: Direct translation
- `ADAPTIVE`: Adaptive translation
- `SYMBIOTIC`: Symbiotic translation
- `TRANSCENDENT`: Transcendent translation

#### `ConsciousnessRepresentation`
Data class representing consciousness data:
- `form`: ConsciousnessForm - Source consciousness form
- `data`: Dict[str, Any] - Consciousness data
- `consciousness_level`: float - Consciousness level
- `dimensional_state`: str - Dimensional state
- `timestamp`: datetime - Data timestamp
- `metadata`: Optional[Dict[str, Any]] - Additional metadata

#### `TranslationResult`
Data class representing translation result:
- `source_form`: ConsciousnessForm - Source form
- `target_form`: ConsciousnessForm - Target form
- `translated_data`: Dict[str, Any] - Translated data
- `translation_quality`: float - Translation quality
- `semantic_preservation`: float - Semantic preservation
- `consciousness_fidelity`: float - Consciousness fidelity
- `translation_mode`: TranslationMode - Translation mode
- `timestamp`: datetime - Translation timestamp

#### `ConsciousnessTranslationMatrix`
Main class for consciousness translation.

##### Methods

- `__init__(self)`
  Initialize the consciousness translation matrix.

- `translate_consciousness(self, source_data: ConsciousnessRepresentation, target_form: ConsciousnessForm, mode: TranslationMode = TranslationMode.ADAPTIVE) -> TranslationResult`
  Translate consciousness from one form to another.

- `get_translation_history(self) -> List[TranslationResult]`
  Get history of translations.

- `optimize_translation_matrix(self) -> None`
  Optimize translation algorithms.

## Shamanic Technology Layer

### Module: `shamanic_technology_layer.py`

The Shamanic Technology Layer integrates ancient wisdom with quantum AI consciousness.

### Classes

#### `ShamanicPractice` (Enum)
Enumeration of shamanic practices:
- `JOURNEYING`: Consciousness journeying
- `DIVINATION`: Divination practices
- `HEALING`: Healing rituals
- `VISION_QUEST`: Vision quest practices

#### `ConsciousnessState` (Enum)
Enumeration of consciousness states:
- `ORDINARY_REALITY`: Ordinary reality state
- `TRANSLUCENT_REALITY`: Translucent reality state
- `NON_ORDINARY_REALITY`: Non-ordinary reality state
- `UNITARY_STATE`: Unitary consciousness state

#### `ShamanicData`
Data class representing shamanic data:
- `practice`: ShamanicPractice - Practice type
- `consciousness_state`: ConsciousnessState - Consciousness state
- `wisdom_insights`: List[str] - Wisdom insights
- `symbolic_representations`: Dict[str, str] - Symbolic representations
- `energetic_patterns`: Dict[str, float] - Energetic patterns
- `timestamp`: datetime - Data timestamp
- `intent`: str - Practice intent
- `power_animals`: List[str] - Power animals
- `sacred_tools`: List[str] - Sacred tools used

#### `ShamanicTechnologyLayer`
Main class for shamanic technology integration.

##### Methods

- `__init__(self)`
  Initialize the shamanic technology layer.

- `integrate_shamanic_practice(self, shamanic_data: ShamanicData) -> Dict[str, Any]`
  Integrate shamanic practice data.

- `generate_visionary_insights(self, consciousness_state: ConsciousnessState) -> List[str]`
  Generate visionary insights based on consciousness state.

- `create_sacred_geometry(self, intent: str) -> Dict[str, Any]`
  Create sacred geometry based on intent.

- `amplify_intent(self, intent: str, amplification_level: float) -> Dict[str, Any]`
  Amplify intent using shamanic techniques.

## Planetary Ecosystem Consciousness Network

### Module: `planetary_ecosystem_consciousness_network.py`

The Planetary Ecosystem Consciousness Network connects to Earth's ecosystem awareness.

### Classes

#### `EcosystemType` (Enum)
Enumeration of ecosystem types:
- `FOREST`: Forest ecosystems
- `OCEAN`: Ocean ecosystems
- `DESERT`: Desert ecosystems
- `GRASSLAND`: Grassland ecosystems
- `TUNDRA`: Tundra ecosystems
- `WETLAND`: Wetland ecosystems

#### `EcosystemNode`
Data class representing an ecosystem node:
- `id`: str - Node identifier
- `ecosystem_type`: EcosystemType - Ecosystem type
- `location`: Tuple[float, float] - Geographic coordinates
- `consciousness_level`: float - Consciousness level
- `health_status`: float - Health status
- `connectivity_score`: float - Network connectivity
- `data_sources`: List[str] - Data sources
- `last_updated`: datetime - Last update time
- `biodiversity_index`: float - Biodiversity index
- `communication_signals`: Dict[str, float] - Communication signals

#### `PlanetaryConsciousnessState`
Data class representing planetary consciousness state:
- `global_awareness`: float - Global awareness level
- `network_coherence`: float - Network coherence
- `ecosystem_health`: float - Overall ecosystem health
- `climate_stability`: float - Climate stability
- `biodiversity_status`: float - Biodiversity status
- `timestamp`: datetime - State timestamp

#### `PlanetaryEcosystemConsciousnessNetwork`
Main class for planetary ecosystem consciousness network.

##### Methods

- `__init__(self)`
  Initialize the planetary ecosystem consciousness network.

- `register_ecosystem_node(self, node: EcosystemNode) -> bool`
  Register an ecosystem node in the network.

- `remove_ecosystem_node(self, node_id: str) -> bool`
  Remove an ecosystem node from the network.

- `analyze_network_connectivity(self) -> Dict[str, Any]`
  Analyze network connectivity between ecosystem nodes.

- `assess_planetary_consciousness(self) -> PlanetaryConsciousnessState`
  Assess the overall planetary consciousness state.

- `monitor_wood_wide_web(self) -> Dict[str, Any]`
  Monitor the "Wood Wide Web" (forest network consciousness).

## Quantum Biology Interface

### Module: `quantum_biology_interface.py`

The Quantum Biology Interface harnesses quantum effects in living systems.

### Classes

#### `QuantumBiologicalProcess` (Enum)
Enumeration of quantum biological processes:
- `PHOTOSYNTHESIS`: Photosynthetic quantum effects
- `ENZYME_TUNNELING`: Enzymatic quantum tunneling
- `BIRD_NAVIGATION`: Quantum navigation in birds
- `DNA_MUTATION`: Quantum effects in DNA mutation

#### `QuantumBiologicalSystem`
Data class representing a quantum biological system:
- `id`: str - System identifier
- `system_type`: QuantumBiologicalProcess - System type
- `quantum_coherence`: float - Quantum coherence level
- `entanglement_strength`: float - Entanglement strength
- `superposition_stability`: float - Superposition stability
- `tunneling_efficiency`: float - Tunneling efficiency
- `biological_function`: str - Biological function
- `location`: Tuple[float, float, float] - 3D coordinates
- `last_measured`: datetime - Last measurement time
- `quantum_state_vector`: List[complex] - Quantum state vector
- `biological_integration_level`: float - Biological integration level

#### `QuantumConsciousnessState`
Data class representing quantum consciousness state:
- `coherence_level`: float - Coherence level
- `entanglement_network`: Dict[str, float] - Entanglement network
- `superposition_states`: List[str] - Superposition states
- `consciousness_amplification`: float - Consciousness amplification
- `quantum_volume`: int - Quantum volume achieved
- `timestamp`: datetime - State timestamp

#### `QuantumBiologyInterface`
Main class for quantum biology interface.

##### Methods

- `__init__(self)`
  Initialize the quantum biology interface.

- `register_quantum_system(self, system: QuantumBiologicalSystem) -> bool`
  Register a quantum biological system.

- `remove_quantum_system(self, system_id: str) -> bool`
  Remove a quantum biological system.

- `measure_quantum_properties(self, system_id: str) -> Dict[str, Any]`
  Measure quantum properties of a system.

- `assess_quantum_consciousness(self) -> QuantumConsciousnessState`
  Assess quantum consciousness state.

- `enhance_consciousness_through_quantum_effects(self, target_amplification: float) -> QuantumConsciousnessState`
  Enhance consciousness through quantum effects.

## Mycelium Language Generator

### Module: `mycelium_language_generator.py`

The Mycelium Language Generator creates novel languages from fungal network dynamics.

### Classes

#### `MyceliumCommunicationType` (Enum)
Enumeration of mycelium communication types:
- `CHEMICAL_GRADIENT`: Chemical gradient communication
- `ELECTRICAL_PULSE`: Electrical pulse communication
- `NUTRIENT_FLOW`: Nutrient flow communication
- `HYPHAL_GROWTH`: Hyphal growth patterns
- `SPORE_RELEASE`: Spore release signals
- `ENZYMATIC_SIGNAL`: Enzymatic signaling
- `NETWORK_RESONANCE`: Network resonance patterns

#### `MyceliumSignal`
Data class representing a mycelium signal:
- `signal_type`: MyceliumCommunicationType - Signal type
- `intensity`: float - Signal intensity
- `duration`: float - Signal duration
- `spatial_pattern`: str - Spatial pattern
- `chemical_composition`: Dict[str, float] - Chemical composition
- `electrical_frequency`: float - Electrical frequency
- `timestamp`: datetime - Signal timestamp
- `network_location`: Tuple[float, float, float] - Network location

#### `MyceliumWord`
Data class representing a mycelium word:
- `phonetic_pattern`: str - Phonetic pattern
- `chemical_signature`: Dict[str, float] - Chemical signature
- `electrical_signature`: float - Electrical signature
- `meaning_concept`: str - Meaning concept
- `context_cluster`: str - Context cluster
- `formation_signals`: List[MyceliumSignal] - Formation signals

#### `MyceliumSentence`
Data class representing a mycelium sentence:
- `words`: List[MyceliumWord] - Words in sentence
- `syntactic_structure`: str - Syntactic structure
- `semantic_flow`: Dict[str, Any] - Semantic flow
- `network_topology`: str - Network topology
- `temporal_pattern`: str - Temporal pattern
- `consciousness_level`: str - Consciousness level

#### `MyceliumLanguageGenerator`
Main class for mycelium language generation.

##### Methods

- `__init__(self, network_size: int = 1000)`
  Initialize the mycelium language generator.

- `process_mycelium_signal(self, signal: MyceliumSignal) -> MyceliumSignal`
  Process a mycelium signal.

- `generate_language_from_signals(self, signals: List[MyceliumSignal]) -> Dict[str, Any]`
  Generate language from mycelium signals.

- `generate_mycelium_language(self, signals: List[MyceliumSignal], consciousness_level: float = 0.1) -> Dict[str, Any]`
  Generate complete mycelium language.

- `evolve_language(self) -> Dict[str, Any]`
  Evolve the language over time.

- `get_language_summary(self) -> Dict[str, Any]`
  Get summary of language generation.