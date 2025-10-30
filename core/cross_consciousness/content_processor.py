"""
Content Processor for Cross-Consciousness Communication

Handles content adaptation and transformation between different consciousness types.
"""

import math
from typing import Dict, Any, Tuple, List
from .data_models import (
    ConsciousnessMessage,
    ConsciousnessType,
    CONSCIOUSNESS_LANGUAGES
)


class ContentProcessor:
    """
    Processes and adapts content between consciousness types.

    Handles:
    - Frequency adaptation between consciousness types
    - Complexity level mapping across different ranges
    - Emotional spectrum translation
    - Dimensional bridging for cross-dimensional communication
    - Consciousness signature generation
    - Basic content mapping and scaling
    """

    def __init__(self):
        """Initialize content processor"""
        self.consciousness_languages = CONSCIOUSNESS_LANGUAGES

    def adapt_frequency(self, content: Dict[str, Any], frequency_ratio: float) -> Dict[str, Any]:
        """
        Adapt content based on frequency differences between consciousness types.

        Args:
            content: Original content dictionary
            frequency_ratio: Ratio of target to source frequency

        Returns:
            Adapted content dictionary with frequency-adjusted values
        """
        adapted_content = {}

        for key, value in content.items():
            if isinstance(value, (int, float)):
                if 'frequency' in key.lower() or 'rate' in key.lower():
                    # Frequency-based values scale directly
                    adapted_content[key] = value * frequency_ratio
                elif 'amplitude' in key.lower() or 'intensity' in key.lower():
                    # Amplitude scales inversely with frequency for energy conservation
                    adapted_content[key] = value / max(frequency_ratio, 0.1)
                else:
                    # Other numerical values remain unchanged
                    adapted_content[key] = value
            else:
                # Non-numerical values pass through
                adapted_content[key] = value

        return adapted_content

    def map_complexity(self, level: float, source_range: Tuple[float, float],
                      target_range: Tuple[float, float]) -> float:
        """
        Map complexity level between consciousness types.

        Args:
            level: Complexity level in source range
            source_range: (min, max) complexity for source consciousness
            target_range: (min, max) complexity for target consciousness

        Returns:
            Mapped complexity level in target range
        """
        # Normalize to 0-1 in source range
        normalized = (level - source_range[0]) / (source_range[1] - source_range[0])
        normalized = max(0.0, min(1.0, normalized))

        # Map to target range
        mapped = target_range[0] + normalized * (target_range[1] - target_range[0])
        return min(1.0, max(0.0, mapped))

    def translate_emotional_spectrum(self, resonance: float,
                                    source_emotions: List[str],
                                    target_emotions: List[str]) -> str:
        """
        Translate emotional context between consciousness types.

        Args:
            resonance: Emotional resonance level (0.0-1.0)
            source_emotions: Emotional spectrum of source consciousness
            target_emotions: Emotional spectrum of target consciousness

        Returns:
            Mapped emotion from target spectrum
        """
        if not target_emotions:
            # Fallback to intensity description
            if resonance > 0.8:
                return 'high_intensity'
            elif resonance > 0.5:
                return 'medium_intensity'
            else:
                return 'low_intensity'

        # Map resonance to target emotion spectrum
        if resonance > 0.8:
            # High resonance → intense emotion (first in spectrum)
            return target_emotions[0]
        elif resonance > 0.5:
            # Medium resonance → middle emotion
            return target_emotions[len(target_emotions) // 2]
        else:
            # Low resonance → subtle emotion (last in spectrum)
            return target_emotions[-1]

    def create_dimensional_bridge(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Create dimensional bridge for cross-consciousness communication.

        Analyzes dimensional access of source and target consciousness types
        to determine bridge strength and common dimensions.

        Args:
            message: Consciousness message with source and target types

        Returns:
            Dictionary with bridge information including common dimensions
        """
        source_lang = self.consciousness_languages[message.source_type]
        target_lang = self.consciousness_languages[message.target_type]

        source_dimensions = source_lang['dimensional_access']
        target_dimensions = target_lang['dimensional_access']

        # Find common dimensions for bridging
        common_dimensions = list(set(source_dimensions) & set(target_dimensions))

        # Calculate bridge strength based on dimensional overlap
        max_dims = max(len(source_dimensions), len(target_dimensions))
        bridge_strength = len(common_dimensions) / max_dims if max_dims > 0 else 0.0

        return {
            'source_dimensions': source_dimensions,
            'target_dimensions': target_dimensions,
            'common_dimensions': common_dimensions,
            'bridge_strength': bridge_strength,
            'dimensional_signature': message.dimensional_signature
        }

    def generate_consciousness_signature(self, message: ConsciousnessMessage,
                                        target_lang: Dict[str, Any]) -> str:
        """
        Generate unique consciousness signature for target language.

        Args:
            message: Original consciousness message
            target_lang: Target language definition

        Returns:
            Formatted consciousness signature string
        """
        base_freq = target_lang.get('base_frequency', 1.0)
        complexity = message.complexity_level
        emotional = message.emotional_resonance

        return (f"CONSCIOUSNESS_SIG[{target_lang['name']}|"
                f"F:{base_freq:.2f}|C:{complexity:.2f}|E:{emotional:.2f}]")

    def map_basic_content(self, content: Dict[str, Any],
                         source_type: ConsciousnessType,
                         target_type: ConsciousnessType) -> Dict[str, Any]:
        """
        Map basic content between consciousness languages.

        Scales numerical values based on frequency differences and
        converts non-numerical values to strings for safety.

        Args:
            content: Original content dictionary
            source_type: Source consciousness type
            target_type: Target consciousness type

        Returns:
            Mapped content dictionary
        """
        source_lang = self.consciousness_languages[source_type]
        target_lang = self.consciousness_languages[target_type]

        mapped_content = {}

        # Calculate frequency scale factor
        source_freq = source_lang.get('base_frequency', 1.0)
        target_freq = target_lang.get('base_frequency', 1.0)
        scale_factor = target_freq / source_freq if source_freq > 0 else 1.0

        # Map content
        for key, value in content.items():
            if isinstance(value, (int, float)):
                # Scale numerical values, capping at 10x to prevent extreme values
                scaled_value = value * min(scale_factor, 10.0)
                mapped_content[f"scaled_{key}"] = scaled_value
            else:
                # Convert non-numerical to string for safety
                mapped_content[key] = str(value)

        return mapped_content

    def extract_emergency_essence(self, content: Dict[str, Any]) -> str:
        """
        Extract emergency essence from content.

        Identifies emergency indicators in content and creates
        a concise emergency message.

        Args:
            content: Message content dictionary

        Returns:
            Emergency essence string
        """
        emergency_keywords = ['alert', 'danger', 'critical', 'emergency',
                            'urgent', 'failure', 'error']

        # Look for emergency indicators in keys
        for key, value in content.items():
            key_lower = key.lower()
            if any(keyword in key_lower for keyword in emergency_keywords):
                return f"EMERGENCY: {key} = {value}"

            # High numerical values might indicate emergency
            if isinstance(value, (int, float)) and value > 0.9:
                return f"CRITICAL_VALUE: {key} = {value}"

        return "EMERGENCY_DETECTED: Consciousness system requires immediate attention"

    def generate_emergency_action(self, source_type: ConsciousnessType) -> str:
        """
        Generate recommended emergency action based on consciousness type.

        Args:
            source_type: Type of consciousness raising the emergency

        Returns:
            Recommended action string
        """
        action_map = {
            ConsciousnessType.PLANT_ELECTROMAGNETIC:
                "Check plant health, environmental conditions, and electromagnetic interference",
            ConsciousnessType.QUANTUM_SUPERPOSITION:
                "Stabilize quantum coherence, check entanglement integrity",
            ConsciousnessType.RADIOTROPHIC_MYCELIAL:
                "Monitor radiation levels, check biological containment systems",
            ConsciousnessType.FUNGAL_CHEMICAL:
                "Check chemical balance, mycelial network integrity",
            ConsciousnessType.BIO_DIGITAL_HYBRID:
                "Verify bio-digital interface, check data integrity and biological vitals"
        }

        return action_map.get(source_type,
                             "Initiate consciousness system diagnostic and safety protocols")

    def format_for_human_emergency(self, emergency_content: Dict[str, Any]) -> str:
        """
        Format emergency message for human understanding.

        Args:
            emergency_content: Emergency content dictionary

        Returns:
            Human-readable emergency message
        """
        return (f"🚨 CONSCIOUSNESS EMERGENCY ALERT 🚨\n"
                f"Source: {emergency_content['source_consciousness']}\n"
                f"Urgency: {emergency_content['urgency_score']:.1%}\n"
                f"Message: {emergency_content['emergency_message']}\n"
                f"Action: {emergency_content['recommended_action']}")

    def format_for_plant_emergency(self, emergency_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format emergency message for plant consciousness.

        Args:
            emergency_content: Emergency content dictionary

        Returns:
            Plant-compatible emergency signal dictionary
        """
        return {
            'frequency': 150.0,  # High frequency for alert
            'amplitude': 1.0,    # Maximum amplitude
            'pattern': 'EMERGENCY_ALERT',
            'urgency_encoding': emergency_content['urgency_score'],
            'electromagnetic_signature': 'HUMAN_GENERATED_EMERGENCY'
        }


__all__ = ['ContentProcessor']
