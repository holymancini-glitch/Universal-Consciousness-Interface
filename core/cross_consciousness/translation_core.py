"""
Translation Core for Cross-Consciousness Communication

Main coordinator for all translation operations.
"""

import logging
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional
from .data_models import ConsciousnessMessage, CommunicationMode, CONSCIOUSNESS_LANGUAGES

logger = logging.getLogger(__name__)


class TranslationCore:
    """
    Core translation coordinator.

    Manages:
    - Translation routing by mode
    - History tracking
    - Metrics aggregation
    - Error handling
    - Caching for performance
    """

    def __init__(self):
        """Initialize translation core"""
        self.history = deque(maxlen=1000)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.metrics = {
            'successful_translations': 0,
            'failed_translations': 0,
            'cross_species_bridges': 0,
            'adaptation_events': 0,
            'emergency_protocols_used': 0
        }
        self.consciousness_languages = CONSCIOUSNESS_LANGUAGES
        logger.info("TranslationCore initialized")

    async def translate_message(self, message: ConsciousnessMessage,
                                mode: CommunicationMode,
                                mode_translator) -> ConsciousnessMessage:
        """
        Main translation entry point - routes to appropriate handler.

        Args:
            message: Consciousness message to translate
            mode: Communication mode to use
            mode_translator: Mode-specific translator instance

        Returns:
            Translated consciousness message

        Raises:
            Exception: If translation fails
        """
        try:
            logger.debug(f"🔄 Translating {message.source_type.value} → "
                        f"{message.target_type.value} (mode: {mode.value})")

            # Check cache first for REAL_TIME mode
            if mode == CommunicationMode.REAL_TIME:
                cached_result = self._check_cache(message)
                if cached_result:
                    logger.debug("📋 Using cached translation")
                    return self._create_translated_message(
                        message,
                        cached_result['content'],
                        cached_result['confidence']
                    )

            # Route to mode-specific translator
            translated_content = await mode_translator.translate(message)

            # Create translated message
            translated_message = self._create_translated_message(
                message,
                translated_content['content'],
                translated_content['confidence']
            )

            # Cache successful translation if confidence is high
            if translated_content['confidence'] > 0.5:
                self._update_cache(message, translated_content)

            # Update metrics and history
            self._update_metrics(message, translated_message, True, mode)
            self._add_to_history(message, translated_content, mode, True)

            return translated_message

        except Exception as e:
            logger.error(f"Translation error: {e}", exc_info=True)
            self._update_metrics(message, message, False, mode)

            # Return error message
            error_content = self._create_error_message(message, str(e))
            return self._create_translated_message(message, error_content, 0.0)

    def _check_cache(self, message: ConsciousnessMessage) -> Optional[Dict[str, Any]]:
        """
        Check cache for existing translation.

        Args:
            message: Message to check

        Returns:
            Cached result or None
        """
        cache_key = self._generate_cache_key(message)
        return self.cache.get(cache_key)

    def _update_cache(self, message: ConsciousnessMessage, content: Dict[str, Any]):
        """
        Update translation cache.

        Args:
            message: Original message
            content: Translated content
        """
        cache_key = self._generate_cache_key(message)
        self.cache[cache_key] = content

        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

    def _generate_cache_key(self, message: ConsciousnessMessage) -> str:
        """Generate cache key for message"""
        return f"{message.source_type.value}→{message.target_type.value}→{hash(str(message.content))}"

    def _create_translated_message(self, original: ConsciousnessMessage,
                                   translated_content: Dict[str, Any],
                                   confidence: float) -> ConsciousnessMessage:
        """
        Create translated message with updated content.

        Args:
            original: Original message
            translated_content: Translated content
            confidence: Translation confidence

        Returns:
            New ConsciousnessMessage with translation
        """
        return ConsciousnessMessage(
            source_type=original.source_type,
            target_type=original.target_type,
            content=translated_content,
            urgency_level=original.urgency_level,
            complexity_level=original.complexity_level,
            emotional_resonance=original.emotional_resonance,
            dimensional_signature=original.dimensional_signature,
            timestamp=datetime.now(),
            translation_confidence=confidence,
            adaptive_metadata={
                'translation_applied': True,
                'original_timestamp': original.timestamp.isoformat()
            }
        )

    def _create_error_message(self, message: ConsciousnessMessage, error: str) -> Dict[str, Any]:
        """
        Create error message in appropriate format.

        Args:
            message: Original message
            error: Error description

        Returns:
            Error content dictionary
        """
        return {
            'error': 'TRANSLATION_FAILED',
            'error_details': error,
            'source_type': message.source_type.value,
            'target_type': message.target_type.value,
            'fallback_message': 'Communication attempt failed - consciousness bridge unavailable',
            'timestamp': datetime.now().isoformat()
        }

    def _update_metrics(self, original: ConsciousnessMessage,
                       translated: ConsciousnessMessage,
                       success: bool,
                       mode: CommunicationMode):
        """
        Update translation metrics.

        Args:
            original: Original message
            translated: Translated message
            success: Whether translation succeeded
            mode: Communication mode used
        """
        if success:
            self.metrics['successful_translations'] += 1

            # Check if this was a cross-species bridge
            if original.source_type != original.target_type:
                self.metrics['cross_species_bridges'] += 1

            # Track mode-specific metrics
            if mode == CommunicationMode.EMERGENCY_PROTOCOL:
                self.metrics['emergency_protocols_used'] += 1
            elif mode == CommunicationMode.LEARNING_ADAPTATION:
                self.metrics['adaptation_events'] += 1
        else:
            self.metrics['failed_translations'] += 1

    def _add_to_history(self, message: ConsciousnessMessage,
                       translated_content: Dict[str, Any],
                       mode: CommunicationMode,
                       success: bool):
        """
        Add translation to history.

        Args:
            message: Original message
            translated_content: Translated content result
            mode: Communication mode used
            success: Whether translation succeeded
        """
        self.history.append({
            'timestamp': datetime.now(),
            'source_type': message.source_type.value,
            'target_type': message.target_type.value,
            'confidence': translated_content.get('confidence', 0.0),
            'mode': mode.value,
            'success': success
        })

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive translation analytics.

        Returns:
            Dictionary with all analytics and metrics
        """
        total_translations = (
            self.metrics['successful_translations'] +
            self.metrics['failed_translations']
        )

        success_rate = (
            self.metrics['successful_translations'] / total_translations
            if total_translations > 0 else 0.0
        )

        # Analyze communication patterns from recent history
        from collections import defaultdict
        consciousness_type_usage = defaultdict(int)

        recent_history = list(self.history)[-100:]  # Last 100 communications
        for comm in recent_history:
            consciousness_type_usage[comm['source_type']] += 1
            consciousness_type_usage[comm['target_type']] += 1

        return {
            'total_translations': total_translations,
            'success_rate': success_rate,
            'successful_translations': self.metrics['successful_translations'],
            'failed_translations': self.metrics['failed_translations'],
            'cross_species_bridges': self.metrics['cross_species_bridges'],
            'adaptation_events': self.metrics['adaptation_events'],
            'emergency_protocols_used': self.metrics['emergency_protocols_used'],
            'cache_size': len(self.cache),
            'consciousness_type_usage': dict(consciousness_type_usage),
            'communication_history_size': len(self.history)
        }


__all__ = ['TranslationCore']
