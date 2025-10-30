"""
Translation Modes for Cross-Consciousness Communication

Implements 5 different translation strategies for different use cases.
"""

import logging
from typing import Dict, Any
from .data_models import (
    ConsciousnessMessage,
    ConsciousnessType,
    CONSCIOUSNESS_LANGUAGES
)
from .content_processor import ContentProcessor
from .pattern_matcher import PatternMatcher
from .bridge_manager import BridgeManager
from .adaptive_learner import AdaptiveLearner

logger = logging.getLogger(__name__)


class RealTimeTranslator:
    """
    Fast rule-based translation for immediate communication.

    Prioritizes speed over depth, using predefined translation rules
    for quick responses.
    """

    def __init__(self, pattern_matcher: PatternMatcher):
        """
        Initialize real-time translator.

        Args:
            pattern_matcher: Pattern matcher with translation rules
        """
        self.pattern_matcher = pattern_matcher
        logger.info("RealTimeTranslator initialized")

    async def translate(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Fast real-time translation for immediate communication.

        Args:
            message: Consciousness message to translate

        Returns:
            Translation result dictionary
        """
        # Find applicable translation rules
        applicable_rules = self.pattern_matcher.find_applicable_rules(
            message.source_type,
            message.target_type,
            message.content
        )

        if not applicable_rules:
            # Fallback to basic pattern-based translation
            logger.debug("No applicable rules found, using pattern-based fallback")
            return await self.pattern_matcher.pattern_based_translation(message)

        # Use best rule based on effectiveness
        best_rule = max(applicable_rules, key=lambda r: r.effectiveness_score)
        translated_content = self.pattern_matcher.apply_rule(best_rule, message.content)

        logger.debug(f"Applied rule: {best_rule.rule_id} (confidence: {best_rule.confidence:.2f})")

        return {
            'content': translated_content,
            'confidence': best_rule.confidence,
            'method': 'rule_based_real_time',
            'rule_used': best_rule.rule_id
        }


class DeepTranslator:
    """
    Deep structural translation with consciousness analysis.

    Analyzes consciousness structures, frequencies, and dimensions
    for comprehensive translation.
    """

    def __init__(self, content_processor: ContentProcessor):
        """
        Initialize deep translator.

        Args:
            content_processor: Content processor for transformations
        """
        self.content_processor = content_processor
        self.consciousness_languages = CONSCIOUSNESS_LANGUAGES
        logger.info("DeepTranslator initialized")

    async def translate(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Deep translation with consciousness structure analysis.

        Args:
            message: Consciousness message to translate

        Returns:
            Translation result dictionary
        """
        source_lang = self.consciousness_languages[message.source_type]
        target_lang = self.consciousness_languages[message.target_type]

        # Frequency adaptation
        frequency_ratio = (
            target_lang['base_frequency'] / source_lang['base_frequency']
            if source_lang['base_frequency'] > 0 else 1.0
        )

        # Complexity level mapping
        complexity_mapping = self.content_processor.map_complexity(
            message.complexity_level,
            source_lang['complexity_range'],
            target_lang['complexity_range']
        )

        # Emotional spectrum translation
        emotional_translation = self.content_processor.translate_emotional_spectrum(
            message.emotional_resonance,
            source_lang['emotional_spectrum'],
            target_lang['emotional_spectrum']
        )

        # Deep structural translation
        translated_content = {
            'original_content': message.content,
            'frequency_adapted': self.content_processor.adapt_frequency(
                message.content, frequency_ratio
            ),
            'complexity_mapped': complexity_mapping,
            'emotional_context': emotional_translation,
            'dimensional_bridge': self.content_processor.create_dimensional_bridge(message),
            'consciousness_signature': self.content_processor.generate_consciousness_signature(
                message, target_lang
            )
        }

        logger.debug(f"Deep translation: {message.source_type.value} → {message.target_type.value} "
                    f"(freq_ratio: {frequency_ratio:.2f})")

        return {
            'content': translated_content,
            'confidence': 0.85,  # High confidence for deep translation
            'method': 'deep_structural',
            'frequency_ratio': frequency_ratio,
            'complexity_mapped': complexity_mapping
        }


class ConsciousnessBridge:
    """
    Three-step bridge-based translation for complex transformations.

    Uses intermediate bridge consciousness state for better translation
    between disparate consciousness types.
    """

    def __init__(self, bridge_manager: BridgeManager):
        """
        Initialize consciousness bridge.

        Args:
            bridge_manager: Bridge manager for bridging operations
        """
        self.bridge_manager = bridge_manager
        logger.info("ConsciousnessBridge initialized")

    async def translate(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Create consciousness bridge for seamless communication.

        Three-step process:
        1. Translate to intermediate bridge state
        2. Enhance in bridge
        3. Translate from bridge to target

        Args:
            message: Consciousness message to translate

        Returns:
            Translation result dictionary
        """
        # Step 1: Create intermediate consciousness bridge
        bridge_type = await self.bridge_manager.create_bridge(
            message.source_type,
            message.target_type
        )

        # Step 2: Translate to bridge consciousness
        bridge_message = await self.bridge_manager.translate_to_bridge(
            message, bridge_type
        )

        # Step 3: Enhance in bridge state
        enhanced_message = await self.bridge_manager.enhance_in_bridge(bridge_message)

        # Step 4: Translate from bridge to target
        final_translation = await self.bridge_manager.translate_from_bridge(
            enhanced_message, message.target_type
        )

        logger.debug(f"Bridge translation: {message.source_type.value} → "
                    f"{bridge_type} → {message.target_type.value}")

        return {
            'content': final_translation,
            'confidence': 0.9,  # Very high confidence
            'method': 'consciousness_bridging',
            'bridge_type': bridge_type,
            'enhancement_applied': True
        }


class EmergencyProtocolTranslator:
    """
    Emergency translation protocol for critical communications.

    Prioritizes clarity and urgency over nuance for emergency situations.
    """

    def __init__(self, content_processor: ContentProcessor):
        """
        Initialize emergency protocol translator.

        Args:
            content_processor: Content processor for emergency formatting
        """
        self.content_processor = content_processor
        logger.info("EmergencyProtocolTranslator initialized")

    async def translate(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Emergency translation protocol for critical communications.

        Args:
            message: Consciousness message (emergency)

        Returns:
            Translation result dictionary
        """
        # Emergency translation prioritizes clarity and urgency
        emergency_essence = self.content_processor.extract_emergency_essence(message.content)
        recommended_action = self.content_processor.generate_emergency_action(message.source_type)

        emergency_content = {
            'alert_level': 'CRITICAL',
            'source_consciousness': message.source_type.value,
            'target_consciousness': message.target_type.value,
            'urgency_score': message.urgency_level,
            'emergency_message': emergency_essence,
            'recommended_action': recommended_action,
            'consciousness_state': 'EMERGENCY_ACTIVE',
            'timestamp': message.timestamp.isoformat()
        }

        # Add target-specific emergency formatting
        if message.target_type == ConsciousnessType.HUMAN_LINGUISTIC:
            emergency_content['human_readable'] = self.content_processor.format_for_human_emergency(
                emergency_content
            )
        elif message.target_type == ConsciousnessType.PLANT_ELECTROMAGNETIC:
            emergency_content['plant_signal'] = self.content_processor.format_for_plant_emergency(
                emergency_content
            )

        logger.warning(f"EMERGENCY TRANSLATION: {message.source_type.value} → "
                      f"{message.target_type.value} (urgency: {message.urgency_level:.2f})")

        return {
            'content': emergency_content,
            'confidence': 1.0,  # Maximum confidence for emergency
            'method': 'emergency_protocol',
            'priority': 'HIGHEST'
        }


class AdaptiveLearningTranslator:
    """
    Learning-based translation that improves over time.

    Uses historical patterns and success metrics to continuously
    improve translation quality.
    """

    def __init__(self, adaptive_learner: AdaptiveLearner):
        """
        Initialize adaptive learning translator.

        Args:
            adaptive_learner: Adaptive learner for pattern analysis
        """
        self.adaptive_learner = adaptive_learner
        self.communication_history = []  # Will be set by TranslationCore
        logger.info("AdaptiveLearningTranslator initialized")

    def set_history(self, history):
        """Set communication history for pattern analysis"""
        self.communication_history = history

    async def translate(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Learning-based translation that improves over time.

        Args:
            message: Consciousness message to translate

        Returns:
            Translation result dictionary
        """
        # Analyze communication patterns
        pattern_analysis = self.adaptive_learner.analyze_communication_patterns(
            message.source_type,
            message.target_type,
            self.communication_history
        )

        # Generate adaptive translation approach
        adaptive_translation = self.adaptive_learner.generate_adaptive_translation(
            message,
            pattern_analysis
        )

        # Estimate translation effectiveness
        effectiveness_score = self.adaptive_learner.estimate_effectiveness(
            adaptive_translation,
            message
        )

        # Learn from this translation if effective
        if effectiveness_score > 0.7:
            self.adaptive_learner.learn_from_success(
                message,
                adaptive_translation,
                effectiveness_score
            )
            logger.debug(f"Learning applied: effectiveness {effectiveness_score:.2f}")

        logger.debug(f"Adaptive translation: {message.source_type.value} → "
                    f"{message.target_type.value} "
                    f"(pattern strength: {pattern_analysis.get('pattern_strength', 0):.2f})")

        return {
            'content': adaptive_translation,
            'confidence': effectiveness_score,
            'method': 'adaptive_learning',
            'learning_applied': effectiveness_score > 0.7,
            'pattern_analysis': pattern_analysis
        }


def get_translator_for_mode(mode_name: str, **dependencies) -> Any:
    """
    Factory function to get appropriate translator for mode.

    Args:
        mode_name: Name of the communication mode
        **dependencies: Required dependencies for each translator

    Returns:
        Translator instance for the mode

    Raises:
        ValueError: If mode_name is unknown
    """
    translators = {
        'REAL_TIME': lambda: RealTimeTranslator(dependencies['pattern_matcher']),
        'DEEP_TRANSLATION': lambda: DeepTranslator(dependencies['content_processor']),
        'CONSCIOUSNESS_BRIDGING': lambda: ConsciousnessBridge(dependencies['bridge_manager']),
        'EMERGENCY_PROTOCOL': lambda: EmergencyProtocolTranslator(dependencies['content_processor']),
        'LEARNING_ADAPTATION': lambda: AdaptiveLearningTranslator(dependencies['adaptive_learner'])
    }

    if mode_name not in translators:
        raise ValueError(f"Unknown translation mode: {mode_name}")

    return translators[mode_name]()


__all__ = [
    'RealTimeTranslator',
    'DeepTranslator',
    'ConsciousnessBridge',
    'EmergencyProtocolTranslator',
    'AdaptiveLearningTranslator',
    'get_translator_for_mode'
]
