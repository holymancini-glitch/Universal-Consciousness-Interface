"""
Performance Assessment for Adaptive Learning System

Assesses various aspects of learning performance including adaptation rate,
error reduction, pattern recognition, and creativity.
"""

import logging
from typing import Any
from datetime import datetime
from collections import deque
import numpy as np

from .data_models import LearningMetrics, LearningPhase

logger = logging.getLogger(__name__)


class PerformanceAssessor:
    """Assesses learning system performance across multiple dimensions"""

    def __init__(self, consciousness_system):
        self.consciousness_system = consciousness_system

    async def assess_learning_performance(
        self,
        learning_history: deque,
        mistake_database: deque,
        creative_solutions: deque,
        current_phase: LearningPhase
    ) -> LearningMetrics:
        """Comprehensive learning performance assessment"""

        # Calculate various learning metrics
        adaptation_rate = self._calculate_adaptation_rate(learning_history)
        error_reduction_rate = self._calculate_error_reduction_rate()
        pattern_recognition_accuracy = self._assess_pattern_recognition()
        creative_generation_score = self._assess_creative_generation(creative_solutions)
        mistake_learning_effectiveness = self._assess_mistake_learning(mistake_database)
        parameter_adaptation_success = self._assess_parameter_adaptation(learning_history)

        # Overall learning efficiency
        overall_efficiency = np.mean([
            adaptation_rate,
            error_reduction_rate,
            pattern_recognition_accuracy,
            creative_generation_score,
            mistake_learning_effectiveness,
            parameter_adaptation_success
        ])

        metrics = LearningMetrics(
            timestamp=datetime.now(),
            learning_phase=current_phase,
            adaptation_rate=adaptation_rate,
            error_reduction_rate=error_reduction_rate,
            pattern_recognition_accuracy=pattern_recognition_accuracy,
            creative_generation_score=creative_generation_score,
            mistake_learning_effectiveness=mistake_learning_effectiveness,
            parameter_adaptation_success=parameter_adaptation_success,
            overall_learning_efficiency=overall_efficiency
        )

        return metrics

    def _calculate_adaptation_rate(self, learning_history: deque) -> float:
        """Calculate how quickly the system adapts to new conditions"""
        if len(learning_history) < 5:
            return 0.5  # Default moderate rate

        # Measure improvement rate over recent learning cycles
        recent_efficiencies = [m.overall_learning_efficiency for m in list(learning_history)[-10:]]

        if len(recent_efficiencies) > 1:
            # Calculate trend (positive = improving)
            trend = np.polyfit(range(len(recent_efficiencies)), recent_efficiencies, 1)[0]
            # Normalize to 0-1 range
            adaptation_rate = min(1.0, max(0.0, 0.5 + trend * 10))
        else:
            adaptation_rate = 0.5

        return adaptation_rate

    def _calculate_error_reduction_rate(self) -> float:
        """Calculate how effectively the system reduces errors over time"""
        feedback_loop = self.consciousness_system.feedback_loop

        if len(feedback_loop.prediction_errors) < 5:
            return 0.3  # Default low rate for new systems

        # Get recent error values
        recent_errors = list(feedback_loop.prediction_errors.values())[-10:]

        if len(recent_errors) > 3:
            # Calculate error reduction trend
            early_errors = np.mean(recent_errors[:3])
            late_errors = np.mean(recent_errors[-3:])

            if early_errors > 0:
                reduction_rate = max(0.0, (early_errors - late_errors) / early_errors)
            else:
                reduction_rate = 0.5
        else:
            reduction_rate = 0.3

        return min(1.0, reduction_rate)

    def _assess_pattern_recognition(self) -> float:
        """Assess pattern recognition capabilities"""
        # Analyze mycelial network's ability to form meaningful patterns
        mycelial_engine = self.consciousness_system.mycelial_engine

        if len(mycelial_engine.experiences) < 10:
            return 0.2  # Low accuracy for small datasets

        # Calculate network efficiency metrics
        graph = mycelial_engine.graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes > 1:
            # Optimal connectivity ratio (not too sparse, not too dense)
            max_edges = num_nodes * (num_nodes - 1) / 2
            connectivity_ratio = num_edges / max_edges if max_edges > 0 else 0

            # Optimal range: 0.1 to 0.4
            if 0.1 <= connectivity_ratio <= 0.4:
                pattern_accuracy = 0.8 + (0.25 - abs(connectivity_ratio - 0.25)) * 0.8
            else:
                pattern_accuracy = max(0.2, 0.8 - abs(connectivity_ratio - 0.25) * 2)
        else:
            pattern_accuracy = 0.2

        return min(1.0, max(0.0, pattern_accuracy))

    def _assess_creative_generation(self, creative_solutions: deque) -> float:
        """Assess creative solution generation capabilities"""
        if len(creative_solutions) < 3:
            return 0.3  # Default moderate creativity for new systems

        # Analyze diversity and quality of creative solutions
        solution_qualities = []
        solution_diversities = []

        for solution in list(creative_solutions)[-10:]:
            quality = solution.get('quality_score', 0.5)
            diversity = solution.get('diversity_score', 0.5)
            solution_qualities.append(quality)
            solution_diversities.append(diversity)

        if solution_qualities and solution_diversities:
            avg_quality = np.mean(solution_qualities)
            avg_diversity = np.mean(solution_diversities)

            # Creative score combines quality and diversity
            creative_score = (avg_quality * 0.6 + avg_diversity * 0.4)
        else:
            creative_score = 0.3

        return min(1.0, max(0.0, creative_score))

    def _assess_mistake_learning(self, mistake_database: deque) -> float:
        """Assess effectiveness of learning from mistakes"""
        if len(mistake_database) < 3:
            return 0.4  # Default moderate effectiveness

        # Analyze how well the system learns from past mistakes
        mistake_reduction_scores = []

        for mistake in list(mistake_database)[-10:]:
            initial_severity = mistake.get('initial_severity', 0.5)
            resolution_effectiveness = mistake.get('resolution_effectiveness', 0.5)
            recurrence_prevention = mistake.get('recurrence_prevention', 0.5)

            # Learning effectiveness combines resolution and prevention
            effectiveness = (resolution_effectiveness * 0.4 + recurrence_prevention * 0.6)
            mistake_reduction_scores.append(effectiveness)

        if mistake_reduction_scores:
            avg_effectiveness = np.mean(mistake_reduction_scores)
        else:
            avg_effectiveness = 0.4

        return min(1.0, max(0.0, avg_effectiveness))

    def _assess_parameter_adaptation(self, learning_history: deque) -> float:
        """Assess how well the system adapts its parameters"""
        if len(learning_history) < 10:
            return 0.5  # Default moderate adaptation

        # Analyze parameter adaptation success over time
        recent_adaptations = [m for m in list(learning_history)[-20:]]

        # Look for improvements in learning efficiency after parameter changes
        adaptation_successes = 0
        total_adaptations = 0

        for i in range(1, len(recent_adaptations)):
            current_efficiency = recent_adaptations[i].overall_learning_efficiency
            previous_efficiency = recent_adaptations[i-1].overall_learning_efficiency

            if current_efficiency > previous_efficiency:
                adaptation_successes += 1
            total_adaptations += 1

        if total_adaptations > 0:
            success_rate = adaptation_successes / total_adaptations
        else:
            success_rate = 0.5

        return success_rate


__all__ = ['PerformanceAssessor']
