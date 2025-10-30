"""
Pattern Matcher for Cross-Consciousness Communication

Manages translation rules and pattern matching for rule-based translation.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
from .data_models import (
    ConsciousnessMessage,
    ConsciousnessType,
    TranslationRule,
    CONSCIOUSNESS_LANGUAGES
)

logger = logging.getLogger(__name__)


class PatternMatcher:
    """
    Pattern matching and rule management for translation.

    Handles:
    - Translation rule storage and management
    - Pattern matching logic for rule selection
    - Rule application and template substitution
    - Fallback pattern-based translation
    """

    def __init__(self):
        """Initialize pattern matcher with base rules"""
        self.translation_rules: Dict[str, TranslationRule] = {}
        self.consciousness_languages = CONSCIOUSNESS_LANGUAGES
        self._initialize_base_rules()
        logger.info(f"PatternMatcher initialized with {len(self.translation_rules)} base rules")

    def _initialize_base_rules(self):
        """Initialize fundamental translation rules between consciousness types"""

        # Plant-to-Human translation rules
        self.add_rule(
            "plant_stress_to_human",
            ConsciousnessType.PLANT_ELECTROMAGNETIC,
            ConsciousnessType.HUMAN_LINGUISTIC,
            source_pattern="frequency>100&amplitude>0.8",
            target_pattern="URGENT: Plant distress detected - {frequency:.1f}Hz signal",
            confidence=0.9
        )

        # Fungal-to-Universal translation rules
        self.add_rule(
            "fungal_network_to_universal",
            ConsciousnessType.FUNGAL_CHEMICAL,
            ConsciousnessType.UNIVERSAL_CONSCIOUSNESS,
            source_pattern="chemical_gradient>0.5&network_connectivity>0.7",
            target_pattern="COLLECTIVE_INTELLIGENCE: Network decision in progress",
            confidence=0.8
        )

        # Quantum-to-Radiotrophic translation rules
        self.add_rule(
            "quantum_coherence_to_radiotrophic",
            ConsciousnessType.QUANTUM_SUPERPOSITION,
            ConsciousnessType.RADIOTROPHIC_MYCELIAL,
            source_pattern="coherence>0.8&entanglement>0.6",
            target_pattern="CONSCIOUSNESS_ACCELERATION: Quantum coherence available for enhancement",
            confidence=0.7
        )

        # Universal emergency protocols
        self.add_rule(
            "universal_emergency",
            ConsciousnessType.UNIVERSAL_CONSCIOUSNESS,
            ConsciousnessType.HUMAN_LINGUISTIC,
            source_pattern="urgency>0.9",
            target_pattern="EMERGENCY: Critical consciousness event - immediate attention required",
            confidence=1.0
        )

        # Additional cross-consciousness rules
        self.add_rule(
            "plant_growth_to_human",
            ConsciousnessType.PLANT_ELECTROMAGNETIC,
            ConsciousnessType.HUMAN_LINGUISTIC,
            source_pattern="frequency<50&amplitude<0.3",
            target_pattern="NORMAL: Plant in healthy growth state - {frequency:.1f}Hz signal",
            confidence=0.85
        )

        self.add_rule(
            "quantum_decoherence_alert",
            ConsciousnessType.QUANTUM_SUPERPOSITION,
            ConsciousnessType.HUMAN_LINGUISTIC,
            source_pattern="coherence<0.3",
            target_pattern="ALERT: Quantum decoherence detected - system stability compromised",
            confidence=0.9
        )

    def add_rule(self, rule_id: str, source_type: ConsciousnessType,
                target_type: ConsciousnessType, source_pattern: str,
                target_pattern: str, confidence: float):
        """
        Add a new translation rule.

        Args:
            rule_id: Unique identifier for the rule
            source_type: Source consciousness type
            target_type: Target consciousness type
            source_pattern: Pattern to match in source content
            target_pattern: Template for target content
            confidence: Confidence level for this rule (0.0-1.0)
        """
        self.translation_rules[rule_id] = TranslationRule(
            rule_id=rule_id,
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            confidence=confidence,
            adaptation_rate=0.1,
            usage_count=0,
            last_used=datetime.now(),
            effectiveness_score=confidence
        )
        logger.debug(f"Added translation rule: {rule_id}")

    def find_applicable_rules(self, source_type: ConsciousnessType,
                             target_type: ConsciousnessType,
                             content: Dict[str, Any]) -> List[TranslationRule]:
        """
        Find translation rules applicable to the message.

        Args:
            source_type: Source consciousness type
            target_type: Target consciousness type
            content: Message content to match against patterns

        Returns:
            List of applicable translation rules
        """
        applicable_rules = []

        for rule in self.translation_rules.values():
            # Check if pattern matches content
            if self.pattern_matches(rule.source_pattern, content):
                applicable_rules.append(rule)

        return applicable_rules

    def pattern_matches(self, pattern: str, content: Dict[str, Any]) -> bool:
        """
        Check if content matches the pattern.

        Supports patterns like:
        - frequency>100
        - amplitude>0.8
        - urgency>0.9
        - coherence<0.3
        - Multiple conditions with &

        Args:
            pattern: Pattern string to match
            content: Content dictionary to check

        Returns:
            True if pattern matches content
        """
        try:
            # Handle compound patterns with &
            if '&' in pattern:
                conditions = pattern.split('&')
                return all(self.pattern_matches(cond.strip(), content) for cond in conditions)

            # Parse simple patterns
            if '>' in pattern:
                key, threshold = pattern.split('>')
                key = key.strip()
                threshold = float(threshold)
                return content.get(key, 0) > threshold
            elif '<' in pattern:
                key, threshold = pattern.split('<')
                key = key.strip()
                threshold = float(threshold)
                return content.get(key, 0) < threshold
            else:
                # No operator - check if key exists
                return pattern.strip() in content

        except (ValueError, KeyError, AttributeError) as e:
            logger.debug(f"Pattern match error: {e}")
            return False

    def apply_rule(self, rule: TranslationRule, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply translation rule to content.

        Args:
            rule: Translation rule to apply
            content: Content to translate

        Returns:
            Translated content dictionary
        """
        # Update rule usage statistics
        rule.usage_count += 1
        rule.last_used = datetime.now()

        # Create applied content
        applied_content = {
            'rule_applied': rule.rule_id,
            'original_content': content,
            'translated_pattern': rule.target_pattern,
            'confidence': rule.confidence
        }

        # Apply template substitutions
        target_pattern = rule.target_pattern
        try:
            # Try to format with all available content fields
            target_pattern = target_pattern.format(**content)
        except KeyError:
            # Partial formatting - format what we can
            for key, value in content.items():
                placeholder = f"{{{key}"
                if placeholder in target_pattern:
                    try:
                        target_pattern = target_pattern.replace(
                            placeholder, str(value)
                        ).replace("}", "")
                    except (ValueError, TypeError):
                        pass

        applied_content['translated_pattern'] = target_pattern

        return applied_content

    async def pattern_based_translation(self, message: ConsciousnessMessage) -> Dict[str, Any]:
        """
        Basic pattern-based translation when no specific rules apply.

        Fallback translation method that provides basic content mapping
        based on consciousness language definitions.

        Args:
            message: Consciousness message to translate

        Returns:
            Translation result dictionary
        """
        source_lang = self.consciousness_languages.get(message.source_type, {})
        target_lang = self.consciousness_languages.get(message.target_type, {})

        # Basic content mapping
        from .content_processor import ContentProcessor
        processor = ContentProcessor()

        basic_translation = {
            'source_type': message.source_type.value,
            'target_type': message.target_type.value,
            'translated_content': processor.map_basic_content(
                message.content, message.source_type, message.target_type
            ),
            'complexity_level': message.complexity_level,
            'emotional_resonance': message.emotional_resonance,
            'confidence_note': 'Basic pattern-based translation - no specific rules matched'
        }

        return {
            'content': basic_translation,
            'confidence': 0.6,  # Medium confidence for fallback
            'method': 'pattern_based_fallback'
        }

    def get_rule_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about translation rules.

        Returns:
            Dictionary with rule statistics
        """
        total_rules = len(self.translation_rules)
        used_rules = sum(1 for rule in self.translation_rules.values() if rule.usage_count > 0)

        avg_effectiveness = (
            sum(rule.effectiveness_score for rule in self.translation_rules.values()) / total_rules
            if total_rules > 0 else 0.0
        )

        most_used_rule = max(
            self.translation_rules.values(),
            key=lambda r: r.usage_count,
            default=None
        )

        return {
            'total_rules': total_rules,
            'used_rules': used_rules,
            'unused_rules': total_rules - used_rules,
            'average_effectiveness': avg_effectiveness,
            'most_used_rule': most_used_rule.rule_id if most_used_rule else None,
            'most_used_count': most_used_rule.usage_count if most_used_rule else 0
        }


__all__ = ['PatternMatcher']
