"""
Adaptive Learner for Cross-Consciousness Communication

Implements learning and adaptation for translation improvement over time.
"""

import logging
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any
from .data_models import ConsciousnessMessage, ConsciousnessType

logger = logging.getLogger(__name__)


class AdaptiveLearner:
    """
    Adaptive learning system for translation improvement.

    Learns from:
    - Successful translations
    - Communication patterns
    - Effectiveness scores
    - Historical performance
    """

    def __init__(self):
        """Initialize adaptive learner"""
        self.adaptive_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.learning_history = deque(maxlen=500)
        self.performance_metrics = {
            'total_adaptations': 0,
            'successful_adaptations': 0,
            'failed_adaptations': 0
        }
        logger.info("AdaptiveLearner initialized")

    def analyze_communication_patterns(self, source_type: ConsciousnessType,
                                      target_type: ConsciousnessType,
                                      history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze communication patterns for learning.

        Args:
            source_type: Source consciousness type
            target_type: Target consciousness type
            history: Communication history list

        Returns:
            Pattern analysis dictionary with metrics
        """
        # Filter relevant history
        relevant_history = [
            comm for comm in history
            if comm.get('source_type') == source_type.value
            and comm.get('target_type') == target_type.value
        ]

        if not relevant_history:
            return {
                'pattern_strength': 0.0,
                'success_rate': 0.0,
                'frequency': 0,
                'recent_trend': 'no_data'
            }

        # Calculate metrics
        success_count = sum(1 for comm in relevant_history if comm.get('success', False))
        success_rate = success_count / len(relevant_history)

        avg_confidence = (
            sum(comm.get('confidence', 0.0) for comm in relevant_history) / len(relevant_history)
        )

        # Determine trend
        if success_rate > 0.8:
            recent_trend = 'improving'
        elif success_rate < 0.5:
            recent_trend = 'needs_attention'
        else:
            recent_trend = 'stable'

        return {
            'pattern_strength': avg_confidence,
            'success_rate': success_rate,
            'frequency': len(relevant_history),
            'recent_trend': recent_trend,
            'avg_confidence': avg_confidence
        }

    def generate_adaptive_translation(self, message: ConsciousnessMessage,
                                     pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate adaptive translation based on learned patterns.

        Args:
            message: Consciousness message to translate
            pattern_analysis: Analysis from analyze_communication_patterns

        Returns:
            Adaptive translation content
        """
        adaptation_strength = pattern_analysis.get('pattern_strength', 0.5)
        success_rate = pattern_analysis.get('success_rate', 0.5)

        # Base adaptive content
        adaptive_content = {
            'base_content': message.content,
            'adaptation_level': adaptation_strength,
            'learning_confidence': success_rate,
            'source_type': message.source_type.value,
            'target_type': message.target_type.value
        }

        # Apply adaptations based on success patterns
        if success_rate > 0.8:
            # High success rate - apply proven optimizations
            adaptive_content['optimization'] = 'proven_successful'
            adaptive_content['enhanced_clarity'] = True
            adaptive_content['confidence_boost'] = 0.2
            logger.debug(f"Applying proven successful pattern (success_rate: {success_rate:.2f})")

        elif success_rate < 0.5:
            # Low success rate - try alternative approach
            adaptive_content['alternative_approach'] = True
            adaptive_content['experimental_translation'] = True
            adaptive_content['confidence_adjustment'] = -0.1
            logger.debug(f"Applying experimental approach (success_rate: {success_rate:.2f})")

        else:
            # Moderate success - standard approach
            adaptive_content['standard_approach'] = True
            logger.debug(f"Applying standard approach (success_rate: {success_rate:.2f})")

        # Add pattern-specific optimizations
        pattern_key = f"{message.source_type.value}_to_{message.target_type.value}"
        if pattern_key in self.adaptive_patterns:
            recent_patterns = self.adaptive_patterns[pattern_key][-5:]  # Last 5 patterns
            if recent_patterns:
                adaptive_content['pattern_history_size'] = len(recent_patterns)
                adaptive_content['pattern_informed'] = True

        return adaptive_content

    def estimate_effectiveness(self, translation: Dict[str, Any],
                              original_message: ConsciousnessMessage) -> float:
        """
        Estimate how effective a translation will be.

        Args:
            translation: Translation content to evaluate
            original_message: Original message being translated

        Returns:
            Effectiveness score (0.0-1.0)
        """
        base_effectiveness = 0.7  # Base effectiveness score

        # Adjust based on translation features
        if translation.get('enhanced_clarity'):
            base_effectiveness += 0.2
        if translation.get('alternative_approach'):
            base_effectiveness += 0.1
        if translation.get('experimental_translation'):
            base_effectiveness -= 0.1  # Experimental approaches are riskier
        if translation.get('pattern_informed'):
            base_effectiveness += 0.15
        if translation.get('optimization') == 'proven_successful':
            base_effectiveness += 0.15

        # Adjust based on message properties
        complexity_penalty = original_message.complexity_level * 0.1
        urgency_adjustment = original_message.urgency_level * 0.05

        final_effectiveness = base_effectiveness - complexity_penalty + urgency_adjustment

        return min(1.0, max(0.0, final_effectiveness))

    def learn_from_success(self, message: ConsciousnessMessage,
                          translation: Dict[str, Any],
                          actual_effectiveness: float = None):
        """
        Learn from successful translation to improve future translations.

        Args:
            message: Original consciousness message
            translation: Successful translation
            actual_effectiveness: Actual measured effectiveness (optional)
        """
        pattern_key = f"{message.source_type.value}_to_{message.target_type.value}"

        # Create learning record
        learning_record = {
            'timestamp': datetime.now(),
            'source_type': message.source_type.value,
            'target_type': message.target_type.value,
            'source_content': message.content,
            'translated_content': translation,
            'complexity_level': message.complexity_level,
            'emotional_resonance': message.emotional_resonance,
            'urgency_level': message.urgency_level,
            'success_indicators': self._extract_success_indicators(translation),
            'actual_effectiveness': actual_effectiveness
        }

        # Store successful pattern
        self.adaptive_patterns[pattern_key].append(learning_record)

        # Limit pattern storage (keep last 50 patterns per type pair)
        if len(self.adaptive_patterns[pattern_key]) > 50:
            self.adaptive_patterns[pattern_key].pop(0)

        # Update learning history
        self.learning_history.append(learning_record)

        # Update metrics
        self.performance_metrics['total_adaptations'] += 1
        self.performance_metrics['successful_adaptations'] += 1

        logger.debug(f"Learned from successful translation: {pattern_key}")

    def _extract_success_indicators(self, translation: Dict[str, Any]) -> List[str]:
        """
        Extract indicators of what made translation successful.

        Args:
            translation: Translation content

        Returns:
            List of success indicator strings
        """
        indicators = []

        if translation.get('enhanced_clarity'):
            indicators.append('enhanced_clarity')
        if translation.get('alternative_approach'):
            indicators.append('alternative_approach')
        if translation.get('optimization'):
            indicators.append(f"optimization_{translation['optimization']}")
        if translation.get('confidence', 0) > 0.8:
            indicators.append('high_confidence')
        if translation.get('pattern_informed'):
            indicators.append('pattern_informed')

        return indicators if indicators else ['standard_success']

    def get_learning_metrics(self) -> Dict[str, Any]:
        """
        Get learning performance metrics.

        Returns:
            Dictionary with learning statistics
        """
        total_patterns = sum(len(patterns) for patterns in self.adaptive_patterns.values())

        success_rate = (
            self.performance_metrics['successful_adaptations'] /
            self.performance_metrics['total_adaptations']
            if self.performance_metrics['total_adaptations'] > 0 else 0.0
        )

        return {
            'total_patterns_learned': total_patterns,
            'pattern_types': len(self.adaptive_patterns),
            'learning_history_size': len(self.learning_history),
            'total_adaptations': self.performance_metrics['total_adaptations'],
            'successful_adaptations': self.performance_metrics['successful_adaptations'],
            'failed_adaptations': self.performance_metrics['failed_adaptations'],
            'adaptation_success_rate': success_rate
        }

    def get_pattern_insights(self, source_type: ConsciousnessType,
                            target_type: ConsciousnessType) -> Dict[str, Any]:
        """
        Get insights about specific consciousness type pair.

        Args:
            source_type: Source consciousness type
            target_type: Target consciousness type

        Returns:
            Dictionary with pattern insights
        """
        pattern_key = f"{source_type.value}_to_{target_type.value}"
        patterns = self.adaptive_patterns.get(pattern_key, [])

        if not patterns:
            return {
                'available': False,
                'message': 'No learning data available for this consciousness pair'
            }

        # Analyze patterns
        avg_complexity = sum(p['complexity_level'] for p in patterns) / len(patterns)
        avg_resonance = sum(p['emotional_resonance'] for p in patterns) / len(patterns)

        # Common success indicators
        all_indicators = []
        for pattern in patterns:
            all_indicators.extend(pattern.get('success_indicators', []))

        from collections import Counter
        common_indicators = Counter(all_indicators).most_common(3)

        return {
            'available': True,
            'total_patterns': len(patterns),
            'avg_complexity': avg_complexity,
            'avg_emotional_resonance': avg_resonance,
            'common_success_indicators': [indicator for indicator, count in common_indicators],
            'latest_pattern_time': patterns[-1]['timestamp'].isoformat() if patterns else None
        }


__all__ = ['AdaptiveLearner']
