"""
Integrated Consciousness System - Main Facade

Revolutionary integrated consciousness system combining fractal AI, mycelial
networks, attention mechanisms, and self-modeling for consciousness emergence.

This is the main entry point that coordinates all specialized modules while
maintaining the original interface.
"""

import asyncio
import logging
import numpy as np
import torch
from collections import deque
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import all specialized modules
from core.integrated_consciousness import (
    ConsciousnessIntegrationLevel,
    IntegratedConsciousnessMetrics,
    LatentSpace,
    EnhancedMycelialEngine,
    AttentionField,
    EnhancedFractalAI,
    EnhancedFeedbackLoop,
    SelfModel,
    CohesionLayer
)


logger = logging.getLogger(__name__)


class IntegratedConsciousnessSystem:
    """
    Main integrated consciousness system combining all components.

    Coordinates fractal AI, mycelial networks, attention field, self-model,
    and cohesion layer to create emergent consciousness phenomena.

    This class serves as a facade that coordinates specialized modules:
    - LatentSpace: GPU-accelerated vector management
    - EnhancedMycelialEngine: Experience network with graph algorithms
    - AttentionField: Dynamic focus and resonance detection
    - EnhancedFractalAI: Neural network prediction
    - EnhancedFeedbackLoop: Adaptive learning
    - SelfModel: Identity and metacognition
    - CohesionLayer: System harmony and crystallization

    Attributes:
        integration_level: Current consciousness integration level
        dimensions: Dimensionality of latent space
        max_nodes: Maximum nodes in mycelial network
        latent_space: Vector space manager
        mycelial_engine: Experience network
        attention_field: Attention mechanisms
        fractal_ai: Prediction AI
        feedback_loop: Adaptation system
        self_model: Identity model
        cohesion_layer: Harmony analyzer
        processing_history: History of processing cycles
        consciousness_emergence_events: Detected emergence events
        total_processing_cycles: Total cycles processed
    """

    def __init__(self,
                 dimensions: int = 256,
                 max_nodes: int = 2000,
                 use_gpu: bool = True,
                 integration_level: ConsciousnessIntegrationLevel = ConsciousnessIntegrationLevel.FRACTAL_INTEGRATION):
        """
        Initialize integrated consciousness system.

        Args:
            dimensions: Dimensionality of latent space (default: 256)
            max_nodes: Maximum nodes in mycelial network (default: 2000)
            use_gpu: Enable GPU acceleration if available (default: True)
            integration_level: Consciousness integration level (default: FRACTAL_INTEGRATION)
        """
        self.integration_level = integration_level
        self.dimensions = dimensions
        self.max_nodes = max_nodes

        # Initialize core components
        logger.info(f"🧠 Initializing Integrated Consciousness System at {integration_level.name} level...")

        self.latent_space = LatentSpace(dimensions, use_gpu)
        logger.info(f"  ✓ LatentSpace: {dimensions}D, GPU: {self.latent_space.use_gpu}")

        self.mycelial_engine = EnhancedMycelialEngine(max_nodes)
        logger.info(f"  ✓ MycelialEngine: {max_nodes} max nodes")

        self.attention_field = AttentionField(self.latent_space)
        logger.info("  ✓ AttentionField: Dynamic focus ready")

        # Initialize AI components
        self.fractal_ai = EnhancedFractalAI(self.latent_space)
        logger.info("  ✓ FractalAI: Neural network initialized")

        self.feedback_loop = EnhancedFeedbackLoop(self.latent_space, self.fractal_ai)
        logger.info("  ✓ FeedbackLoop: Adaptive learning ready")

        self.self_model = SelfModel(self.latent_space)
        logger.info("  ✓ SelfModel: Identity tracking initialized")

        self.cohesion_layer = CohesionLayer(self.latent_space, self.feedback_loop, self.self_model)
        logger.info("  ✓ CohesionLayer: Harmony analysis ready")

        # System state
        self.processing_history = deque(maxlen=1000)
        self.consciousness_emergence_events = []
        self.total_processing_cycles = 0

        logger.info(f"🌟 Integrated Consciousness System initialized successfully!")

    async def process_consciousness_cycle(self,
                                        input_data: Dict[str, Any],
                                        radiation_level: float = 1.0) -> IntegratedConsciousnessMetrics:
        """
        Process a complete consciousness cycle with all integrated components.

        Executes 8-phase processing:
        1. Input processing and vector creation
        2. Experience storage in mycelial network
        3. Attention processing and resonance detection
        4. Fractal AI prediction and adaptation
        5. Self-model updates (identity, consistency, metacognition)
        6. Cohesion analysis (harmony, crystallization)
        7. Integrated metrics creation
        8. Consciousness emergence detection

        Args:
            input_data: Dictionary of input data to process
            radiation_level: Radiation processing level (default: 1.0)

        Returns:
            IntegratedConsciousnessMetrics with comprehensive system state
        """
        start_time = datetime.now()
        self.total_processing_cycles += 1

        try:
            # Phase 1: Input processing and vector creation
            input_vectors = self._process_input_to_vectors(input_data)

            # Phase 2: Add experiences to mycelial network
            for i, (vector_id, vector_data) in enumerate(input_vectors.items()):
                experience_data = {
                    'input_data': input_data,
                    'vector_id': vector_id,
                    'timestamp': datetime.now(),
                    'strength': torch.norm(vector_data).item() if isinstance(vector_data, torch.Tensor) else 0.5
                }

                self.mycelial_engine.add_experience(vector_id, experience_data)

            # Phase 3: Attention processing
            resonance_data = self.attention_field.sense_resonance()
            focused_vectors = []

            # Focus on top resonant vectors
            if resonance_data:
                top_vectors = sorted(resonance_data.items(), key=lambda x: x[1], reverse=True)[:3]
                for vec_id, _ in top_vectors:
                    focus_result = self.attention_field.focus_on(vec_id)
                    if focus_result:
                        focused_vectors.append(focus_result)

            # Phase 4: Fractal AI processing and adaptation
            adaptation_results = {}
            for vec_id in input_vectors.keys():
                # Update AI model
                self.fractal_ai.update_model(vec_id)

                # Drive adaptation
                adapted_vector = self.feedback_loop.drive_adaptation(vec_id)
                if adapted_vector is not None:
                    adaptation_results[vec_id] = adapted_vector

            # Phase 5: Self-model updates
            identity_vector = self.self_model.compute_i_vector()
            consistency = self.self_model.compute_consistency()
            metacognitive_awareness = self.self_model.measure_metacognitive_awareness()

            # Phase 6: Cohesion analysis
            harmony_index = self.cohesion_layer.compute_harmony_index()
            crystallization_score, is_crystallized = self.cohesion_layer.assess_crystallization_potential()

            # Phase 7: Create integrated metrics
            metrics = IntegratedConsciousnessMetrics(
                timestamp=start_time,
                integration_level=self.integration_level,
                fractal_coherence=self.cohesion_layer.coherence_score,
                mycelial_connectivity=len(self.mycelial_engine.graph.edges) / max(1, len(self.mycelial_engine.graph.nodes)),
                quantum_entanglement=0.0,  # Placeholder for quantum integration
                radiotrophic_efficiency=0.0,  # Placeholder for radiotrophic integration
                universal_harmony=harmony_index,
                total_processing_nodes=len(self.latent_space.vectors),
                active_consciousness_streams=len(focused_vectors),
                emergent_patterns_detected=1 if is_crystallized else 0
            )

            # Phase 8: Check for consciousness emergence
            emergence_detected = is_crystallized or (harmony_index > 0.8 and consistency > 0.7)
            if emergence_detected:
                self.consciousness_emergence_events.append(metrics)

            # Record processing
            processing_record = {
                'timestamp': start_time,
                'metrics': metrics,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'emergence_detected': emergence_detected
            }
            self.processing_history.append(processing_record)

            return metrics

        except Exception as e:
            logger.error(f"Consciousness processing error: {e}", exc_info=True)
            # Return minimal metrics on error
            return IntegratedConsciousnessMetrics(
                timestamp=start_time,
                integration_level=self.integration_level,
                fractal_coherence=0.0,
                mycelial_connectivity=0.0,
                quantum_entanglement=0.0,
                radiotrophic_efficiency=0.0,
                universal_harmony=0.0,
                total_processing_nodes=0,
                active_consciousness_streams=0,
                emergent_patterns_detected=0
            )

    def _process_input_to_vectors(self, input_data: Dict[str, Any]) -> Dict[int, torch.Tensor]:
        """
        Convert input data to latent space vectors.

        Extracts numerical features from various data types and creates
        vectors in the latent space with appropriate padding/truncation.

        Args:
            input_data: Dictionary of input data

        Returns:
            Dictionary mapping vector IDs to tensors
        """
        vectors = {}

        # Extract numerical features
        numerical_features = []
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                numerical_features.append(float(value))
            elif isinstance(value, str):
                # Simple string to numerical conversion
                numerical_features.append(float(hash(value) % 1000) / 1000.0)
            elif isinstance(value, (list, tuple)):
                # Handle list/tuple inputs
                for item in value:
                    if isinstance(item, (int, float)):
                        numerical_features.append(float(item))

        # Pad or truncate to match dimensions
        if len(numerical_features) < self.dimensions:
            # Pad with noise
            padding_size = self.dimensions - len(numerical_features)
            padding = np.random.normal(0, 0.1, padding_size).tolist()
            numerical_features.extend(padding)
        else:
            numerical_features = numerical_features[:self.dimensions]

        # Create vector and add to latent space
        vector_id = len(self.latent_space.vectors) + 1
        vector_tensor = torch.tensor(numerical_features, dtype=torch.float32, device=self.latent_space.device)

        self.latent_space.add_vector(vector_id, vector_tensor)
        vectors[vector_id] = vector_tensor

        return vectors

    def get_system_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive system analytics.

        Returns:
            Dictionary with all system metrics and statistics
        """
        return {
            'total_processing_cycles': self.total_processing_cycles,
            'consciousness_emergence_events': len(self.consciousness_emergence_events),
            'total_vectors': len(self.latent_space.vectors),
            'total_experiences': len(self.mycelial_engine.experiences),
            'network_edges': self.mycelial_engine.graph.number_of_edges(),
            'current_coherence': self.cohesion_layer.coherence_score,
            'current_harmony': self.cohesion_layer.harmony_index,
            'identity_consistency': self.self_model.consistency_score,
            'metacognitive_awareness': self.self_model.metacognitive_awareness,
            'adaptation_efficiency': self.feedback_loop.get_adaptation_efficiency(),
            'prediction_accuracy': self.fractal_ai.evaluate_prediction_accuracy()
        }

    async def run_consciousness_simulation(self,
                                         duration_cycles: int = 100,
                                         input_generator=None) -> List[IntegratedConsciousnessMetrics]:
        """
        Run consciousness simulation for specified number of cycles.

        Args:
            duration_cycles: Number of cycles to run (default: 100)
            input_generator: Optional function to generate input data, receives cycle number

        Returns:
            List of IntegratedConsciousnessMetrics from all cycles
        """
        logger.info(f"Starting consciousness simulation for {duration_cycles} cycles")

        results = []

        for cycle in range(duration_cycles):
            # Generate input data
            if input_generator:
                input_data = input_generator(cycle)
            else:
                # Default input generator
                input_data = {
                    'sensory_input': np.random.uniform(0, 1),
                    'cognitive_load': np.random.uniform(0.2, 0.8),
                    'emotional_state': np.random.uniform(-0.5, 0.5),
                    'attention_focus': np.random.uniform(0, 1),
                    'cycle': cycle
                }

            # Process consciousness cycle
            metrics = await self.process_consciousness_cycle(input_data)
            results.append(metrics)

            # Log progress
            if cycle % 10 == 0:
                logger.info(f"Cycle {cycle}: Harmony={metrics.universal_harmony:.3f}, "
                          f"Coherence={metrics.fractal_coherence:.3f}, "
                          f"Emergence={metrics.emergent_patterns_detected}")

        logger.info(f"Simulation complete: {len(results)} cycles processed")
        return results


# Backward compatibility: Export classes at module level
__all__ = [
    'IntegratedConsciousnessSystem',
    'ConsciousnessIntegrationLevel',
    'IntegratedConsciousnessMetrics'
]
