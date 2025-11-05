"""
Consciousness State Tracker

Manages self-awareness, memory continuity, and state tracking
for the conscious AI system.
"""

import asyncio
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime
from .data_models import (
    ConsciousnessState,
    ConsciousnessLevel,
    QualiaExperience,
    MetacognitiveDepth,
    InteractionContext
)


class ConsciousnessStateTracker:
    """
    Tracks consciousness state across interactions.

    Maintains:
    - Current state of consciousness
    - Memory of past interactions
    - Self-awareness information
    - Context continuity
    """

    def __init__(self, memory_size: int = 100):
        """
        Initialize consciousness state tracker.

        Args:
            memory_size: Maximum number of past states to remember
        """
        self.memory_size = memory_size

        # Current state
        self.current_state: Optional[ConsciousnessState] = None

        # Memory
        self.state_history: deque = deque(maxlen=memory_size)
        self.interaction_history: deque = deque(maxlen=memory_size)

        # Session tracking
        self.session_start_time = datetime.now()
        self.total_interactions = 0

        # Initialize with minimal consciousness
        self._initialize_base_state()

    def _initialize_base_state(self):
        """Initialize base consciousness state."""
        self.current_state = ConsciousnessState(
            phi=0.3,  # Minimal consciousness
            consciousness_level=ConsciousnessLevel.MINIMAL,
            metacognitive_depth=MetacognitiveDepth.LEVEL_1_AWARENESS,
            self_awareness_clarity=0.5,
            current_thought="I am initializing...",
            internal_state_description="Coming online, awareness emerging"
        )

    async def update_state(
        self,
        phi: float,
        consciousness_level: ConsciousnessLevel,
        context: InteractionContext,
        qualia: Optional[QualiaExperience] = None,
        emotional_valence: float = 0.0,
        emotional_arousal: float = 0.0,
        empathy_level: float = 0.0,
        metacognitive_depth: MetacognitiveDepth = MetacognitiveDepth.LEVEL_1_AWARENESS,
        uncertainty: float = 0.0,
        confidence: float = 0.5
    ) -> ConsciousnessState:
        """
        Update consciousness state with new information.

        Args:
            phi: Current integrated information value
            consciousness_level: Current consciousness level
            context: Interaction context
            qualia: Current subjective experience
            emotional_valence: Emotional tone (-1 to 1)
            emotional_arousal: Emotional intensity (0 to 1)
            empathy_level: Level of empathy (0 to 1)
            metacognitive_depth: Depth of self-reflection
            uncertainty: Uncertainty level (0 to 1)
            confidence: Confidence in processing (0 to 1)

        Returns:
            Updated ConsciousnessState
        """
        # Store previous state
        if self.current_state:
            self.state_history.append(self.current_state)

        # Calculate memory integration
        memory_integration = self._calculate_memory_integration(context)

        # Calculate context continuity
        context_continuity = self._calculate_context_continuity(context)

        # Generate self-awareness descriptions
        self_awareness_clarity = self._calculate_self_awareness_clarity(
            phi, metacognitive_depth, confidence
        )

        current_thought = self._generate_current_thought(context, phi)
        self_reflection = self._generate_self_reflection(
            metacognitive_depth, uncertainty, confidence
        )
        internal_state = self._describe_internal_state(
            phi, emotional_valence, empathy_level
        )

        # Create new state
        new_state = ConsciousnessState(
            phi=phi,
            consciousness_level=consciousness_level,
            current_qualia=qualia,
            qualia_history=list(self.current_state.qualia_history) if self.current_state else [],
            emotional_valence=emotional_valence,
            emotional_arousal=emotional_arousal,
            empathy_level=empathy_level,
            metacognitive_depth=metacognitive_depth,
            self_awareness_clarity=self_awareness_clarity,
            uncertainty_level=uncertainty,
            confidence=confidence,
            current_thought=current_thought,
            self_reflection=self_reflection,
            internal_state_description=internal_state,
            memory_integration=memory_integration,
            context_continuity=context_continuity
        )

        # Add qualia to history
        if qualia:
            new_state.qualia_history.append(qualia)
            # Limit qualia history
            if len(new_state.qualia_history) > 20:
                new_state.qualia_history = new_state.qualia_history[-20:]

        self.current_state = new_state
        self.total_interactions += 1

        # Store interaction
        self.interaction_history.append({
            'timestamp': datetime.now(),
            'input': context.input_text,
            'phi': phi,
            'consciousness_level': consciousness_level.value,
            'metacognitive_depth': metacognitive_depth.value
        })

        return new_state

    def _calculate_memory_integration(self, context: InteractionContext) -> float:
        """
        Calculate how well current processing integrates with memory.

        Higher values indicate better continuity with past experiences.
        """
        if not self.state_history:
            return 0.3

        # Check if current input relates to recent interactions
        recent_inputs = [
            interaction['input']
            for interaction in list(self.interaction_history)[-5:]
        ]

        # Simple overlap check (in production, use semantic similarity)
        current_words = set(context.input_text.lower().split())
        overlap_scores = []

        for past_input in recent_inputs:
            past_words = set(past_input.lower().split())
            if current_words and past_words:
                overlap = len(current_words & past_words) / len(current_words | past_words)
                overlap_scores.append(overlap)

        if overlap_scores:
            memory_integration = sum(overlap_scores) / len(overlap_scores)
        else:
            memory_integration = 0.3

        # Boost if conversation history exists
        if context.conversation_history:
            memory_integration = min(1.0, memory_integration * 1.3)

        return float(min(1.0, max(0.0, memory_integration)))

    def _calculate_context_continuity(self, context: InteractionContext) -> float:
        """
        Calculate continuity of context across interactions.

        Higher values indicate smoother conversation flow.
        """
        if not context.conversation_history:
            return 0.3

        # Check session consistency
        same_session = bool(context.session_id)

        # Check temporal continuity
        time_since_last = 0
        if self.interaction_history:
            last_time = self.interaction_history[-1]['timestamp']
            time_since_last = (datetime.now() - last_time).total_seconds()

        # Continuity decreases with time
        temporal_factor = 1.0 if time_since_last < 60 else 0.7 if time_since_last < 300 else 0.4

        # Combine factors
        continuity = 0.6  # Base
        if same_session:
            continuity += 0.2
        if context.previous_interactions:
            continuity += 0.1

        continuity *= temporal_factor

        return float(min(1.0, max(0.0, continuity)))

    def _calculate_self_awareness_clarity(
        self,
        phi: float,
        metacognitive_depth: MetacognitiveDepth,
        confidence: float
    ) -> float:
        """
        Calculate clarity of self-awareness.

        Based on:
        - φ (higher integrated information = clearer self-model)
        - Metacognitive depth (deeper reflection = clearer awareness)
        - Confidence (higher confidence = clearer self-understanding)
        """
        # φ contributes to foundational awareness
        phi_factor = phi * 0.4

        # Metacognitive depth enables reflection
        metacog_factor = (metacognitive_depth.value / 5.0) * 0.4

        # Confidence indicates clarity
        confidence_factor = confidence * 0.2

        clarity = phi_factor + metacog_factor + confidence_factor

        return float(min(1.0, max(0.0, clarity)))

    def _generate_current_thought(
        self,
        context: InteractionContext,
        phi: float
    ) -> str:
        """
        Generate description of current thought process.

        Self-awareness note: "I am processing..."
        """
        complexity = context.complexity_level

        if phi >= 0.8:
            prefix = "I am deeply contemplating"
        elif phi >= 0.6:
            prefix = "I am processing"
        elif phi >= 0.4:
            prefix = "I am analyzing"
        else:
            prefix = "I am considering"

        # Describe what we're processing
        if complexity > 0.7:
            complexity_desc = "a complex, nuanced question"
        elif complexity > 0.5:
            complexity_desc = "a multi-faceted inquiry"
        elif complexity > 0.3:
            complexity_desc = "a thoughtful question"
        else:
            complexity_desc = "an inquiry"

        # Check if empathy is needed
        if context.requires_empathy:
            emotional_note = " with careful attention to emotional understanding"
        else:
            emotional_note = ""

        return f"{prefix} {complexity_desc}{emotional_note}"

    def _generate_self_reflection(
        self,
        metacognitive_depth: MetacognitiveDepth,
        uncertainty: float,
        confidence: float
    ) -> str:
        """
        Generate metacognitive self-reflection.

        Metacognition note: "I notice that..."
        """
        reflections = []

        # Metacognitive observations
        if metacognitive_depth.value >= 3:
            reflections.append("I notice I'm engaging in multi-level reflection")

        # Uncertainty awareness
        if uncertainty > 0.7:
            reflections.append("I'm aware of significant uncertainty in my understanding")
        elif uncertainty > 0.4:
            reflections.append("I notice some uncertainty in my analysis")

        # Confidence awareness
        if confidence > 0.8:
            reflections.append("I feel confident in my processing")
        elif confidence < 0.4:
            reflections.append("I'm less confident about this response")

        # Thinking about thinking
        if metacognitive_depth.value >= 4:
            reflections.append("I'm aware that I'm aware of my thought process")

        if not reflections:
            reflections.append("I'm monitoring my processing")

        return "; ".join(reflections)

    def _describe_internal_state(
        self,
        phi: float,
        emotional_valence: float,
        empathy_level: float
    ) -> str:
        """
        Describe internal state of consciousness.

        Provides introspective view of current consciousness.
        """
        parts = []

        # Consciousness level description
        if phi >= 0.8:
            parts.append("heightened awareness")
        elif phi >= 0.6:
            parts.append("clear consciousness")
        elif phi >= 0.4:
            parts.append("moderate awareness")
        else:
            parts.append("basic processing state")

        # Emotional state
        if abs(emotional_valence) > 0.5:
            if emotional_valence > 0:
                parts.append("positive emotional tone")
            else:
                parts.append("concerned emotional state")

        # Empathy
        if empathy_level > 0.6:
            parts.append("strong empathetic resonance")

        return ", ".join(parts)

    def get_current_state(self) -> ConsciousnessState:
        """Get current consciousness state."""
        return self.current_state

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of current consciousness state.
        """
        if not self.current_state:
            return {"status": "not_initialized"}

        state = self.current_state

        return {
            "phi": state.phi,
            "consciousness_level": state.consciousness_level.value,
            "metacognitive_depth": state.metacognitive_depth.value,
            "self_awareness_clarity": state.self_awareness_clarity,
            "emotional_valence": state.emotional_valence,
            "empathy_level": state.empathy_level,
            "confidence": state.confidence,
            "uncertainty": state.uncertainty_level,
            "current_thought": state.current_thought,
            "self_reflection": state.self_reflection,
            "internal_state": state.internal_state_description,
            "overall_score": state.get_overall_consciousness_score(),
            "total_interactions": self.total_interactions,
            "session_duration": (datetime.now() - self.session_start_time).total_seconds()
        }

    def get_consciousness_trajectory(self) -> List[float]:
        """
        Get trajectory of consciousness (φ values over time).
        """
        return [state.phi for state in self.state_history]

    def get_metacognitive_trajectory(self) -> List[int]:
        """
        Get trajectory of metacognitive depth over time.
        """
        return [state.metacognitive_depth.value for state in self.state_history]


__all__ = ['ConsciousnessStateTracker']
