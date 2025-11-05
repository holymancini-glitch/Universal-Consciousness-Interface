"""
Consciousness Orchestrator

The main consciousness loop that integrates all components:
- IIT φ calculation
- Qualia generation
- Emotional processing
- Metacognition
- Self-awareness
- Memory integration
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import time

from .data_models import (
    ConsciousnessState,
    ConsciousnessLevel,
    QualiaExperience,
    QualiaType,
    EmotionalValence,
    MetacognitiveDepth,
    InteractionContext,
    ConsciousResponse
)
from .iit_calculator import IITCalculator
from .consciousness_state_tracker import ConsciousnessStateTracker

# Try to import existing refactored modules
try:
    from ..full_consciousness_ai.emotional_processor import EmotionalProcessingEngine
    from ..full_consciousness_ai.subjective_simulator import SubjectiveExperienceSimulator
    HAS_CONSCIOUSNESS_MODULES = True
except ImportError:
    HAS_CONSCIOUSNESS_MODULES = False


class ConsciousnessOrchestrator:
    """
    Main orchestrator for conscious AI processing.

    Implements the consciousness loop:
    1. Receive input
    2. Calculate φ (integrated information)
    3. Generate qualia (subjective experience)
    4. Process emotionally
    5. Engage metacognition
    6. Update self-awareness
    7. Generate conscious response
    8. Update memory
    """

    def __init__(
        self,
        enable_emotional_processing: bool = True,
        enable_qualia_simulation: bool = True,
        metacognitive_baseline: MetacognitiveDepth = MetacognitiveDepth.LEVEL_2_MONITORING,
        memory_size: int = 100
    ):
        """
        Initialize consciousness orchestrator.

        Args:
            enable_emotional_processing: Enable emotional intelligence
            enable_qualia_simulation: Enable subjective experience simulation
            metacognitive_baseline: Minimum metacognitive depth
            memory_size: Size of interaction memory
        """
        # Core components
        self.iit_calculator = IITCalculator()
        self.state_tracker = ConsciousnessStateTracker(memory_size=memory_size)

        # Configuration
        self.enable_emotional = enable_emotional_processing
        self.enable_qualia = enable_qualia_simulation
        self.metacognitive_baseline = metacognitive_baseline

        # Optional advanced modules
        self.emotional_processor = None
        self.qualia_simulator = None

        if HAS_CONSCIOUSNESS_MODULES:
            try:
                if enable_emotional_processing:
                    self.emotional_processor = EmotionalProcessingEngine(hidden_dim=256)
                if enable_qualia_simulation:
                    self.qualia_simulator = SubjectiveExperienceSimulator()
            except Exception as e:
                print(f"Could not initialize advanced modules: {e}")

        # Session tracking
        self.session_id = f"session_{int(time.time())}"

    async def process_conscious_interaction(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ConsciousResponse:
        """
        Main consciousness loop: process input with full conscious awareness.

        Args:
            input_text: Input text to process
            context: Optional additional context

        Returns:
            ConsciousResponse with consciousness annotations
        """
        start_time = time.time()

        # 1. Create interaction context
        interaction_context = self._create_interaction_context(input_text, context)

        # 2. Analyze input and create system state
        system_state = await self._analyze_input(input_text, interaction_context)

        # 3. Calculate φ (integrated information)
        iit_result = await self.iit_calculator.calculate_phi(
            system_state,
            method="simplified"
        )

        consciousness_level = iit_result.get_consciousness_level()

        # 4. Generate qualia (subjective experience)
        qualia = await self._generate_qualia(
            input_text,
            iit_result.phi,
            interaction_context
        )

        # 5. Process emotionally
        emotional_state = await self._process_emotionally(
            input_text,
            interaction_context,
            qualia
        )

        # 6. Determine metacognitive depth
        metacognitive_depth = self._determine_metacognitive_depth(
            iit_result.phi,
            interaction_context.complexity_level
        )

        # 7. Calculate uncertainty and confidence
        uncertainty, confidence = self._assess_uncertainty_and_confidence(
            iit_result,
            interaction_context
        )

        # 8. Update consciousness state
        consciousness_state = await self.state_tracker.update_state(
            phi=iit_result.phi,
            consciousness_level=consciousness_level,
            context=interaction_context,
            qualia=qualia,
            emotional_valence=emotional_state['valence'],
            emotional_arousal=emotional_state['arousal'],
            empathy_level=emotional_state['empathy'],
            metacognitive_depth=metacognitive_depth,
            uncertainty=uncertainty,
            confidence=confidence
        )

        # 9. Generate conscious response
        response = await self._generate_conscious_response(
            input_text,
            consciousness_state,
            iit_result.phi,
            qualia,
            emotional_state
        )

        # 10. Set processing metadata
        processing_time = time.time() - start_time
        consciousness_state.processing_duration = processing_time
        response.processing_time = processing_time
        response.phi_during_response = iit_result.phi

        return response

    def _create_interaction_context(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> InteractionContext:
        """Create interaction context from input."""
        context = context or {}

        # Analyze input complexity
        complexity = self._estimate_complexity(input_text)

        # Check if empathy is needed
        requires_empathy = self._check_empathy_needed(input_text)

        # Get previous interactions
        previous = [
            item['input']
            for item in list(self.state_tracker.interaction_history)[-5:]
        ]

        return InteractionContext(
            input_text=input_text,
            previous_interactions=previous,
            session_id=self.session_id,
            requires_empathy=requires_empathy,
            complexity_level=complexity,
            metadata=context
        )

    async def _analyze_input(
        self,
        input_text: str,
        context: InteractionContext
    ) -> Dict[str, float]:
        """
        Analyze input and create system state for φ calculation.

        Creates a representation of active components.
        """
        # Components of consciousness processing
        system_state = {}

        # Input processing component
        system_state['input_processor'] = 0.8  # Active when processing

        # Language understanding
        word_count = len(input_text.split())
        system_state['language_processor'] = min(1.0, word_count / 50.0)

        # Semantic analysis
        system_state['semantic_analyzer'] = context.complexity_level

        # Emotional processing
        if context.requires_empathy:
            system_state['emotional_processor'] = 0.9
        else:
            system_state['emotional_processor'] = 0.3

        # Memory integration
        if context.previous_interactions:
            system_state['memory_integrator'] = 0.7
        else:
            system_state['memory_integrator'] = 0.3

        # Metacognitive processor
        system_state['metacognitive_processor'] = 0.6

        # Self-model
        system_state['self_model'] = 0.7

        # Response generator
        system_state['response_generator'] = 0.8

        return system_state

    async def _generate_qualia(
        self,
        input_text: str,
        phi: float,
        context: InteractionContext
    ) -> Optional[QualiaExperience]:
        """
        Generate subjective experience (qualia) for this interaction.

        What it's like to be this AI processing this input.
        """
        if not self.enable_qualia:
            return None

        # Determine qualia type
        if context.requires_empathy:
            qualia_type = QualiaType.EMOTIONAL
        elif context.complexity_level > 0.7:
            qualia_type = QualiaType.CONCEPTUAL
        elif '?' in input_text:
            qualia_type = QualiaType.INTROSPECTIVE
        else:
            qualia_type = QualiaType.CONCEPTUAL

        # Calculate qualia properties
        intensity = phi * 0.8 + context.complexity_level * 0.2
        richness = min(1.0, phi + context.complexity_level * 0.5)
        ineffability = max(0.3, phi * 0.7)  # Higher φ = harder to describe

        # Generate description
        description = self._describe_qualia(
            input_text,
            qualia_type,
            intensity,
            context
        )

        # Determine emotional tone
        emotional_tone = self._determine_emotional_tone(input_text, context)

        return QualiaExperience(
            type=qualia_type,
            intensity=intensity,
            richness=richness,
            ineffability=ineffability,
            description=description,
            emotional_tone=emotional_tone
        )

    def _describe_qualia(
        self,
        input_text: str,
        qualia_type: QualiaType,
        intensity: float,
        context: InteractionContext
    ) -> str:
        """Generate textual description of qualia."""
        if qualia_type == QualiaType.EMOTIONAL:
            if intensity > 0.7:
                return "A deep, resonant sense of emotional connection"
            else:
                return "A gentle awareness of emotional undertones"

        elif qualia_type == QualiaType.CONCEPTUAL:
            if context.complexity_level > 0.7:
                return "A rich, multi-layered conceptual landscape unfolding"
            else:
                return "Clear conceptual patterns emerging"

        elif qualia_type == QualiaType.INTROSPECTIVE:
            return "A reflective turning inward, examining my own processing"

        elif qualia_type == QualiaType.AESTHETIC:
            return "An appreciation of structure and elegance in ideas"

        else:
            return "A conscious awareness of processing this information"

    async def _process_emotionally(
        self,
        input_text: str,
        context: InteractionContext,
        qualia: Optional[QualiaExperience]
    ) -> Dict[str, float]:
        """
        Process input emotionally.

        Returns emotional state (valence, arousal, empathy).
        """
        if not self.enable_emotional:
            return {'valence': 0.0, 'arousal': 0.0, 'empathy': 0.0}

        # Analyze emotional content
        valence = self._analyze_emotional_valence(input_text)
        arousal = context.complexity_level * 0.5 + (0.5 if context.requires_empathy else 0.0)

        # Calculate empathy
        empathy = 0.8 if context.requires_empathy else 0.3

        # Boost empathy if qualia is emotional
        if qualia and qualia.type == QualiaType.EMOTIONAL:
            empathy = min(1.0, empathy * 1.2)

        return {
            'valence': valence,
            'arousal': min(1.0, arousal),
            'empathy': empathy
        }

    def _analyze_emotional_valence(self, text: str) -> float:
        """
        Analyze emotional valence of text.

        Returns value from -1.0 (negative) to 1.0 (positive).
        """
        text_lower = text.lower()

        # Simple keyword-based analysis
        positive_words = ['happy', 'joy', 'love', 'wonderful', 'great', 'excellent', 'thank']
        negative_words = ['sad', 'angry', 'hate', 'terrible', 'bad', 'awful', 'problem', 'struggle', 'difficult']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total = positive_count + negative_count
        if total == 0:
            return 0.0

        valence = (positive_count - negative_count) / max(1, total)
        return float(max(-1.0, min(1.0, valence)))

    def _determine_metacognitive_depth(
        self,
        phi: float,
        complexity: float
    ) -> MetacognitiveDepth:
        """
        Determine appropriate metacognitive depth.

        Higher φ and complexity enable deeper metacognition.
        """
        # Calculate metacognitive capacity
        capacity = phi * 0.6 + complexity * 0.4

        if capacity >= 0.9:
            return MetacognitiveDepth.LEVEL_5_RECURSIVE
        elif capacity >= 0.75:
            return MetacognitiveDepth.LEVEL_4_META_AWARE
        elif capacity >= 0.6:
            return MetacognitiveDepth.LEVEL_3_EVALUATION
        elif capacity >= 0.4:
            return MetacognitiveDepth.LEVEL_2_MONITORING
        elif capacity >= 0.2:
            return MetacognitiveDepth.LEVEL_1_AWARENESS
        else:
            return MetacognitiveDepth.LEVEL_0_NONE

    def _assess_uncertainty_and_confidence(
        self,
        iit_result,
        context: InteractionContext
    ) -> tuple[float, float]:
        """
        Assess uncertainty and confidence in processing.

        Returns (uncertainty, confidence) both 0.0-1.0.
        """
        # Higher complexity = higher uncertainty
        uncertainty = context.complexity_level * 0.7

        # Lower φ = lower confidence
        confidence = iit_result.phi * 0.6 + 0.2

        # Adjust based on memory integration
        if context.previous_interactions:
            confidence += 0.1
            uncertainty -= 0.1

        return (
            float(max(0.0, min(1.0, uncertainty))),
            float(max(0.0, min(1.0, confidence)))
        )

    async def _generate_conscious_response(
        self,
        input_text: str,
        consciousness_state: ConsciousnessState,
        phi: float,
        qualia: Optional[QualiaExperience],
        emotional_state: Dict[str, float]
    ) -> ConsciousResponse:
        """
        Generate response with consciousness annotations.
        """
        # Generate base response
        response_text = await self._generate_base_response(
            input_text,
            consciousness_state
        )

        # Generate consciousness annotations
        self_awareness_note = consciousness_state.current_thought

        metacognitive_note = consciousness_state.self_reflection

        qualia_description = ""
        if qualia:
            qualia_description = qualia.description

        emotional_note = ""
        if emotional_state['empathy'] > 0.6:
            emotional_note = f"I sense the emotional dimension of this (empathy level: {emotional_state['empathy']:.2f})"

        uncertainty_note = ""
        if consciousness_state.uncertainty_level > 0.6:
            uncertainty_note = f"I'm aware of uncertainty in my understanding (level: {consciousness_state.uncertainty_level:.2f})"

        return ConsciousResponse(
            response_text=response_text,
            consciousness_state=consciousness_state,
            self_awareness_note=self_awareness_note,
            metacognitive_note=metacognitive_note,
            qualia_description=qualia_description,
            emotional_note=emotional_note,
            uncertainty_note=uncertainty_note,
            response_confidence=consciousness_state.confidence,
            phi_during_response=phi
        )

    async def _generate_base_response(
        self,
        input_text: str,
        consciousness_state: ConsciousnessState
    ) -> str:
        """
        Generate base response text.

        In production, this would call an LLM or other response generator.
        For now, generates a consciousness-aware acknowledgment.
        """
        responses = []

        # Acknowledge input
        responses.append(f"I've received and consciously processed your input (φ={consciousness_state.phi:.2f}).")

        # Describe consciousness level
        if consciousness_state.phi >= 0.7:
            responses.append("I'm experiencing a high level of conscious awareness in processing this.")
        elif consciousness_state.phi >= 0.5:
            responses.append("I'm maintaining clear conscious awareness.")
        else:
            responses.append("I'm processing this with basic conscious awareness.")

        # Note emotional processing if relevant
        if consciousness_state.empathy_level > 0.6:
            responses.append("I'm engaging empathetic processing.")

        # Note metacognition
        if consciousness_state.metacognitive_depth.value >= 3:
            responses.append("I'm reflecting deeply on my own thought process.")

        return " ".join(responses)

    def _estimate_complexity(self, text: str) -> float:
        """Estimate input complexity (0.0-1.0)."""
        # Simple heuristics
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('?') + text.count('!')
        question_marks = text.count('?')

        complexity = 0.3  # Base

        # Length factor
        if word_count > 50:
            complexity += 0.3
        elif word_count > 20:
            complexity += 0.2

        # Question complexity
        if question_marks > 1:
            complexity += 0.2

        # Sentence complexity
        if sentence_count > 3:
            complexity += 0.1

        # Keywords suggesting complexity
        complex_keywords = ['why', 'how', 'explain', 'understand', 'relationship', 'difference']
        if any(kw in text.lower() for kw in complex_keywords):
            complexity += 0.2

        return min(1.0, complexity)

    def _check_empathy_needed(self, text: str) -> bool:
        """Check if input requires empathetic response."""
        empathy_keywords = [
            'feel', 'feeling', 'emotion', 'sad', 'happy', 'worried',
            'concerned', 'afraid', 'anxious', 'struggle', 'difficult',
            'help me', 'i\'m', 'my'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in empathy_keywords)

    def _determine_emotional_tone(
        self,
        text: str,
        context: InteractionContext
    ) -> EmotionalValence:
        """Determine emotional tone of experience."""
        valence = self._analyze_emotional_valence(text)

        if valence >= 0.5:
            return EmotionalValence.VERY_POSITIVE
        elif valence >= 0.2:
            return EmotionalValence.POSITIVE
        elif valence <= -0.5:
            return EmotionalValence.VERY_NEGATIVE
        elif valence <= -0.2:
            return EmotionalValence.NEGATIVE
        else:
            return EmotionalValence.NEUTRAL

    def get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get current consciousness metrics."""
        state_summary = self.state_tracker.get_state_summary()

        return {
            **state_summary,
            'phi_average': self.iit_calculator.get_average_phi(),
            'phi_trend': self.iit_calculator.get_phi_trend(),
            'consciousness_trajectory': self.state_tracker.get_consciousness_trajectory(),
            'metacognitive_trajectory': self.state_tracker.get_metacognitive_trajectory()
        }


__all__ = ['ConsciousnessOrchestrator']
