"""
Bridge Manager for Cross-Consciousness Communication

Manages consciousness bridging for seamless cross-consciousness communication.
"""

import logging
from typing import Dict, Any
from .data_models import ConsciousnessType, ConsciousnessMessage, CONSCIOUSNESS_LANGUAGES
from .content_processor import ContentProcessor

logger = logging.getLogger(__name__)


class BridgeManager:
    """
    Manages consciousness bridging for seamless communication.

    Implements three-step bridging process:
    1. Translate source → intermediate bridge
    2. Enhance in bridge state
    3. Translate bridge → target

    The bridge acts as an intermediate consciousness state that
    facilitates better translation between disparate consciousness types.
    """

    def __init__(self):
        """Initialize bridge manager"""
        self.consciousness_languages = CONSCIOUSNESS_LANGUAGES
        self.content_processor = ContentProcessor()
        self.bridge_cache: Dict[str, Dict[str, Any]] = {}
        logger.info("BridgeManager initialized")

    async def create_bridge(self, source_type: ConsciousnessType,
                          target_type: ConsciousnessType) -> str:
        """
        Create intermediate consciousness bridge between source and target.

        Selects optimal bridge type based on consciousness compatibility
        and dimensional overlap.

        Args:
            source_type: Source consciousness type
            target_type: Target consciousness type

        Returns:
            Bridge type identifier string
        """
        bridge_key = f"{source_type.value}_to_{target_type.value}"

        # Check cache first
        if bridge_key in self.bridge_cache:
            return self.bridge_cache[bridge_key]['bridge_type']

        # Select optimal bridge
        bridge_type = self._select_optimal_bridge(source_type, target_type)

        # Cache bridge configuration
        self.bridge_cache[bridge_key] = {
            'bridge_type': bridge_type,
            'source': source_type.value,
            'target': target_type.value
        }

        logger.debug(f"Created bridge: {bridge_type}")
        return bridge_type

    def _select_optimal_bridge(self, source: ConsciousnessType,
                              target: ConsciousnessType) -> str:
        """
        Select optimal bridge consciousness type.

        Strategy:
        - For biological ↔ digital: Use BIO_DIGITAL_HYBRID
        - For quantum ↔ biological: Use UNIVERSAL_CONSCIOUSNESS
        - For physical ↔ abstract: Use ECOSYSTEM_HARMONIC
        - Default: Create combined bridge name

        Args:
            source: Source consciousness type
            target: Target consciousness type

        Returns:
            Optimal bridge type identifier
        """
        # Special case bridges
        biological_types = {
            ConsciousnessType.PLANT_ELECTROMAGNETIC,
            ConsciousnessType.FUNGAL_CHEMICAL,
            ConsciousnessType.RADIOTROPHIC_MYCELIAL,
            ConsciousnessType.ANIMAL_BEHAVIORAL
        }

        quantum_types = {
            ConsciousnessType.QUANTUM_SUPERPOSITION,
            ConsciousnessType.PSYCHOACTIVE_DIMENSIONAL
        }

        # Bio-digital bridge
        if source == ConsciousnessType.BIO_DIGITAL_HYBRID or target == ConsciousnessType.BIO_DIGITAL_HYBRID:
            return "bio_digital_hybrid_bridge"

        # Universal consciousness bridge for quantum/biological
        if (source in quantum_types and target in biological_types) or \
           (target in quantum_types and source in biological_types):
            return "universal_consciousness_bridge"

        # Ecosystem harmonic bridge for multi-organism communication
        if source in biological_types and target in biological_types:
            return "ecosystem_harmonic_bridge"

        # Default combined bridge
        return f"bridge_{source.value}_to_{target.value}"

    async def translate_to_bridge(self, message: ConsciousnessMessage,
                                 bridge_type: str) -> Dict[str, Any]:
        """
        Translate message to bridge consciousness state (Step 1).

        Args:
            message: Original consciousness message
            bridge_type: Type of bridge to translate to

        Returns:
            Bridge-state message dictionary
        """
        # Create dimensional bridge analysis
        dimensional_bridge = self.content_processor.create_dimensional_bridge(message)

        bridge_content = {
            'bridge_type': bridge_type,
            'original_source': message.source_type.value,
            'original_target': message.target_type.value,
            'bridge_content': message.content,
            'dimensional_analysis': dimensional_bridge,
            'urgency_level': message.urgency_level,
            'complexity_level': message.complexity_level,
            'emotional_resonance': message.emotional_resonance,
            'dimensional_signature': message.dimensional_signature
        }

        logger.debug(f"Translated to bridge: {bridge_type}")
        return bridge_content

    async def enhance_in_bridge(self, bridge_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance message while in bridge state (Step 2).

        Bridge enhancement includes:
        - Dimensional alignment optimization
        - Content normalization for target
        - Emotional resonance amplification
        - Complexity smoothing

        Args:
            bridge_message: Message in bridge state

        Returns:
            Enhanced bridge message
        """
        enhanced = bridge_message.copy()

        # Apply bridge enhancements
        enhancements = {
            'enhanced': True,
            'dimensional_alignment': self._calculate_alignment(bridge_message),
            'content_normalized': True,
            'resonance_amplified': bridge_message.get('emotional_resonance', 0.5) * 1.2,
            'complexity_smoothed': True
        }

        # Smooth complexity (reduce extreme values)
        original_complexity = bridge_message.get('complexity_level', 0.5)
        enhanced['complexity_level'] = 0.5 + (original_complexity - 0.5) * 0.8

        # Amplify emotional resonance (but cap at 1.0)
        original_resonance = bridge_message.get('emotional_resonance', 0.5)
        enhanced['emotional_resonance'] = min(1.0, original_resonance * 1.15)

        enhanced['bridge_enhancements'] = enhancements

        logger.debug("Bridge enhancement applied")
        return enhanced

    def _calculate_alignment(self, bridge_message: Dict[str, Any]) -> float:
        """
        Calculate dimensional alignment score.

        Args:
            bridge_message: Message in bridge state

        Returns:
            Alignment score (0.0-1.0)
        """
        dimensional_analysis = bridge_message.get('dimensional_analysis', {})
        bridge_strength = dimensional_analysis.get('bridge_strength', 0.5)

        # Alignment is based on bridge strength and content complexity
        complexity = bridge_message.get('complexity_level', 0.5)

        # Higher bridge strength and moderate complexity = better alignment
        alignment = bridge_strength * (1.0 - abs(complexity - 0.5))

        return max(0.0, min(1.0, alignment))

    async def translate_from_bridge(self, enhanced_message: Dict[str, Any],
                                   target_type: ConsciousnessType) -> Dict[str, Any]:
        """
        Translate from bridge to target consciousness (Step 3).

        Args:
            enhanced_message: Enhanced message in bridge state
            target_type: Target consciousness type

        Returns:
            Final translated content for target consciousness
        """
        target_lang = self.consciousness_languages.get(target_type, {})

        # Extract bridge content
        bridge_content = enhanced_message.get('bridge_content', {})
        enhancements = enhanced_message.get('bridge_enhancements', {})

        # Calculate final confidence based on bridge quality
        alignment = enhancements.get('dimensional_alignment', 0.5)
        base_confidence = 0.9  # High confidence for bridge-based translation

        final_confidence = base_confidence * (0.7 + 0.3 * alignment)

        # Create final translated content
        final_content = {
            'final_content': bridge_content,
            'target_type': target_type.value,
            'target_optimized': True,
            'bridge_enhanced': True,
            'dimensional_alignment': alignment,
            'consciousness_signature': self.content_processor.generate_consciousness_signature(
                ConsciousnessMessage(
                    source_type=ConsciousnessType.UNIVERSAL_CONSCIOUSNESS,  # Bridge source
                    target_type=target_type,
                    content=bridge_content,
                    urgency_level=enhanced_message.get('urgency_level', 0.5),
                    complexity_level=enhanced_message.get('complexity_level', 0.5),
                    emotional_resonance=enhanced_message.get('emotional_resonance', 0.5),
                    dimensional_signature=enhanced_message.get('dimensional_signature', 'bridge'),
                    timestamp=enhanced_message.get('timestamp', None)
                ),
                target_lang
            ),
            'bridge_confidence': final_confidence
        }

        logger.debug(f"Translated from bridge to {target_type.value}")
        return final_content

    def get_bridge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about bridge usage.

        Returns:
            Dictionary with bridge statistics
        """
        return {
            'total_bridges_created': len(self.bridge_cache),
            'bridge_types': list(set(b['bridge_type'] for b in self.bridge_cache.values())),
            'cached_bridges': len(self.bridge_cache)
        }


__all__ = ['BridgeManager']
