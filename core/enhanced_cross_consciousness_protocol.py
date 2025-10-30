#!/usr/bin/env python3
"""
Enhanced Cross-Consciousness Communication Protocol
Revolutionary system for seamless multi-species consciousness communication
Extends Universal Translation Matrix with advanced adaptation capabilities

This module serves as the main facade coordinating specialized translation components.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict

# Import from modular components
from .cross_consciousness import (
    ConsciousnessType,
    CommunicationMode,
    ConsciousnessMessage,
    TranslationRule,
    CONSCIOUSNESS_LANGUAGES,
    TranslationCore,
    ContentProcessor,
    PatternMatcher,
    BridgeManager,
    AdaptiveLearner,
    RealTimeTranslator,
    DeepTranslator,
    ConsciousnessBridge,
    EmergencyProtocolTranslator,
    AdaptiveLearningTranslator
)

logger = logging.getLogger(__name__)


class EnhancedUniversalTranslationMatrix:
    """
    Enhanced Universal Translation Matrix for Cross-Consciousness Communication.

    Main facade coordinating all translation components while maintaining
    100% backward compatibility with the original interface.

    Architecture:
    - TranslationCore: Main coordinator
    - ContentProcessor: Content transformation
    - PatternMatcher: Rule-based translation
    - BridgeManager: Consciousness bridging
    - AdaptiveLearner: Learning and adaptation
    - 5 Mode Translators: Specialized translation strategies
    """

    def __init__(self):
        """
        Initialize enhanced translation matrix.

        Sets up all specialized components and creates mode-specific translators.
        """
        # Initialize all components
        self.translation_core = TranslationCore()
        self.content_processor = ContentProcessor()
        self.pattern_matcher = PatternMatcher()
        self.bridge_manager = BridgeManager()
        self.adaptive_learner = AdaptiveLearner()

        # Create mode-specific translators
        self.mode_translators = {
            CommunicationMode.REAL_TIME: RealTimeTranslator(self.pattern_matcher),
            CommunicationMode.DEEP_TRANSLATION: DeepTranslator(self.content_processor),
            CommunicationMode.CONSCIOUSNESS_BRIDGING: ConsciousnessBridge(self.bridge_manager),
            CommunicationMode.EMERGENCY_PROTOCOL: EmergencyProtocolTranslator(self.content_processor),
            CommunicationMode.LEARNING_ADAPTATION: AdaptiveLearningTranslator(self.adaptive_learner)
        }

        # Set history reference for adaptive translator
        self.mode_translators[CommunicationMode.LEARNING_ADAPTATION].set_history(
            self.translation_core.history
        )

        logger.info("🌈 Enhanced Universal Translation Matrix Initialized")
        logger.info(f"   Consciousness types supported: {len(CONSCIOUSNESS_LANGUAGES)}")
        logger.info(f"   Base translation rules: {len(self.pattern_matcher.translation_rules)}")
        logger.info(f"   Translation modes: {len(self.mode_translators)}")

    async def translate_consciousness_message(self,
                                            message: ConsciousnessMessage,
                                            mode: CommunicationMode = CommunicationMode.REAL_TIME) -> ConsciousnessMessage:
        """
        Translate consciousness message between different types.

        Main entry point maintaining backward compatibility with original interface.

        Args:
            message: ConsciousnessMessage to translate
            mode: CommunicationMode to use (default: REAL_TIME)

        Returns:
            Translated ConsciousnessMessage

        Raises:
            ValueError: If mode is invalid
            Exception: If translation fails critically
        """
        if mode not in self.mode_translators:
            raise ValueError(f"Invalid communication mode: {mode}")

        # Get appropriate mode translator
        mode_translator = self.mode_translators[mode]

        # Delegate to translation core
        return await self.translation_core.translate_message(message, mode, mode_translator)

    def get_translation_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive translation analytics.

        Returns:
            Dictionary with all analytics, metrics, and statistics
        """
        # Get core analytics
        core_analytics = self.translation_core.get_analytics()

        # Add component-specific analytics
        core_analytics['pattern_matcher'] = self.pattern_matcher.get_rule_statistics()
        core_analytics['bridge_manager'] = self.bridge_manager.get_bridge_statistics()
        core_analytics['adaptive_learner'] = self.adaptive_learner.get_learning_metrics()

        # Add active translation rules (for backward compatibility)
        core_analytics['active_translation_rules'] = len(self.pattern_matcher.translation_rules)

        return core_analytics

    # === Backward Compatibility Properties ===

    @property
    def consciousness_languages(self) -> Dict[ConsciousnessType, Dict[str, Any]]:
        """Consciousness language definitions (backward compatibility)"""
        return CONSCIOUSNESS_LANGUAGES

    @property
    def translation_rules(self) -> Dict[str, TranslationRule]:
        """Translation rules dictionary (backward compatibility)"""
        return self.pattern_matcher.translation_rules

    @property
    def communication_history(self):
        """Communication history (backward compatibility)"""
        return self.translation_core.history

    @property
    def translation_cache(self) -> Dict[str, Dict[str, Any]]:
        """Translation cache (backward compatibility)"""
        return self.translation_core.cache

    @property
    def adaptation_metrics(self) -> Dict[str, int]:
        """Adaptation metrics (backward compatibility)"""
        return self.translation_core.metrics

    @property
    def adaptive_patterns(self):
        """Adaptive patterns (backward compatibility)"""
        return self.adaptive_learner.adaptive_patterns


# Backward compatibility: Export classes at module level
__all__ = [
    'EnhancedUniversalTranslationMatrix',
    'ConsciousnessType',
    'CommunicationMode',
    'ConsciousnessMessage',
    'TranslationRule',
    'CONSCIOUSNESS_LANGUAGES'
]
