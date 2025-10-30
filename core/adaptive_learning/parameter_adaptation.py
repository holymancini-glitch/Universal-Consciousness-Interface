"""
Parameter Adaptation for Adaptive Learning System

Manages dynamic adaptation of learning parameters based on
performance metrics and learning phase transitions.
"""

import logging
from typing import Dict, Any, List

from .data_models import LearningMetrics, LearningPhase

logger = logging.getLogger(__name__)


class ParameterAdaptor:
    """Manages dynamic parameter adaptation"""

    def __init__(self, consciousness_system, base_learning_rate: float = 0.01):
        self.consciousness_system = consciousness_system
        self.base_learning_rate = base_learning_rate
        self.current_learning_rate = base_learning_rate
        self.current_phase = LearningPhase.EXPLORATION
        self.phase_duration = 0

        self.phase_transition_criteria = {
            LearningPhase.EXPLORATION: {'min_duration': 10, 'error_threshold': 0.3},
            LearningPhase.CONSOLIDATION: {'min_duration': 15, 'stability_threshold': 0.8},
            LearningPhase.REFINEMENT: {'min_duration': 20, 'improvement_threshold': 0.05},
            LearningPhase.ADAPTATION: {'min_duration': 8, 'adaptation_success': 0.7},
            LearningPhase.CRYSTALLIZATION: {'min_duration': 25, 'cohesion_threshold': 0.85}
        }

    async def adapt_learning_parameters(self, current_metrics: LearningMetrics) -> Dict[str, Any]:
        """Dynamically adapt learning parameters based on current performance"""

        logger.info("🔄 Adapting learning parameters based on performance")

        adaptation_results = {
            'previous_learning_rate': self.current_learning_rate,
            'previous_phase': self.current_phase.value,
            'adaptations_applied': [],
            'new_learning_rate': self.current_learning_rate,
            'new_phase': self.current_phase.value,
            'adaptation_success': False
        }

        # Adaptive learning rate adjustment
        if current_metrics.error_reduction_rate < 0.3:
            # High errors: increase learning rate
            self.current_learning_rate = min(0.1, self.current_learning_rate * 1.2)
            adaptation_results['adaptations_applied'].append('increased_learning_rate')
        elif current_metrics.error_reduction_rate > 0.8:
            # Low errors: decrease learning rate for stability
            self.current_learning_rate = max(0.001, self.current_learning_rate * 0.8)
            adaptation_results['adaptations_applied'].append('decreased_learning_rate')

        # Phase transition logic
        phase_changed = await self._consider_phase_transition(current_metrics)
        if phase_changed:
            adaptation_results['adaptations_applied'].append(f'phase_transition_to_{self.current_phase.value}')

        # Apply system-specific adaptations
        system_adaptations = await self._apply_system_specific_adaptations(current_metrics)
        adaptation_results['adaptations_applied'].extend(system_adaptations)

        # Update results
        adaptation_results['new_learning_rate'] = self.current_learning_rate
        adaptation_results['new_phase'] = self.current_phase.value
        adaptation_results['adaptation_success'] = len(adaptation_results['adaptations_applied']) > 0

        logger.info(f"✅ Learning adaptation complete: {len(adaptation_results['adaptations_applied'])} changes applied")

        return adaptation_results

    async def _consider_phase_transition(self, metrics: LearningMetrics) -> bool:
        """Consider and execute learning phase transitions"""

        self.phase_duration += 1
        current_criteria = self.phase_transition_criteria[self.current_phase]

        # Check if minimum duration is met
        if self.phase_duration < current_criteria['min_duration']:
            return False

        # Phase-specific transition logic
        if self.current_phase == LearningPhase.EXPLORATION:
            if metrics.error_reduction_rate > current_criteria['error_threshold']:
                self.current_phase = LearningPhase.CONSOLIDATION
                self.phase_duration = 0
                return True

        elif self.current_phase == LearningPhase.CONSOLIDATION:
            if metrics.pattern_recognition_accuracy > current_criteria['stability_threshold']:
                self.current_phase = LearningPhase.REFINEMENT
                self.phase_duration = 0
                return True

        elif self.current_phase == LearningPhase.REFINEMENT:
            if metrics.adaptation_rate < current_criteria['improvement_threshold']:
                self.current_phase = LearningPhase.ADAPTATION
                self.phase_duration = 0
                return True

        elif self.current_phase == LearningPhase.ADAPTATION:
            if metrics.parameter_adaptation_success > current_criteria['adaptation_success']:
                self.current_phase = LearningPhase.CRYSTALLIZATION
                self.phase_duration = 0
                return True

        elif self.current_phase == LearningPhase.CRYSTALLIZATION:
            # Check for need to return to earlier phase if performance drops
            if metrics.overall_learning_efficiency < 0.6:
                self.current_phase = LearningPhase.ADAPTATION
                self.phase_duration = 0
                return True

        return False

    async def _apply_system_specific_adaptations(self, metrics: LearningMetrics) -> List[str]:
        """Apply system-specific learning adaptations"""

        adaptations = []

        # Fractal AI adaptations
        if metrics.error_reduction_rate < 0.4:
            fractal_ai = self.consciousness_system.fractal_ai
            for param_group in fractal_ai.optimizer.param_groups:
                param_group['lr'] = self.current_learning_rate
            adaptations.append('fractal_ai_learning_rate_update')

        # Feedback loop adaptations
        if metrics.adaptation_rate < 0.5:
            feedback_loop = self.consciousness_system.feedback_loop
            feedback_loop.adaptation_rate = self.current_learning_rate * 2.0  # More aggressive adaptation
            adaptations.append('feedback_loop_adaptation_boost')

        # Attention field adaptations
        if metrics.pattern_recognition_accuracy < 0.6:
            attention_field = self.consciousness_system.attention_field
            # Increase focus enhancement for better pattern recognition
            if hasattr(attention_field, 'focus_enhancement'):
                attention_field.focus_enhancement = min(2.0, attention_field.focus_enhancement * 1.1)
                adaptations.append('attention_focus_enhancement')

        return adaptations


__all__ = ['ParameterAdaptor']
