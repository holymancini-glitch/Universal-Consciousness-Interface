"""
Adaptive Learning System Core

Main coordination class for the advanced adaptive learning system
with multi-dimensional learning capabilities.
"""

import asyncio
import logging
import torch.nn as nn
from typing import Dict, List, Any
from collections import deque
import numpy as np

from .data_models import LearningMetrics, LearningPhase
from .performance_assessment import PerformanceAssessor
from .parameter_adaptation import ParameterAdaptor
from .mistake_learning import MistakeLearner
from .creative_engine import CreativeEngine

logger = logging.getLogger(__name__)


class AdaptiveLearningSystem:
    """Advanced adaptive learning system with multi-dimensional learning capabilities"""

    def __init__(self, consciousness_system):
        self.consciousness_system = consciousness_system
        self.learning_history = deque(maxlen=200)
        self.mistake_database = deque(maxlen=100)
        self.pattern_library = {}
        self.creative_solutions = deque(maxlen=50)

        # Adaptive learning parameters
        self.base_learning_rate = 0.01
        self.current_learning_rate = self.base_learning_rate
        self.learning_momentum = 0.9
        self.adaptation_threshold = 0.1

        # Create specialized components
        self.performance_assessor = PerformanceAssessor(consciousness_system)
        self.parameter_adaptor = ParameterAdaptor(consciousness_system, self.base_learning_rate)
        self.mistake_learner = MistakeLearner(self.base_learning_rate)
        self.creative_engine = CreativeEngine()

        # Advanced learning mechanisms (legacy compatibility)
        self.meta_learning_network = self._build_meta_learning_network()
        self.creativity_engine = self.creative_engine.creativity_network
        self.mistake_analyzer = self._build_mistake_analyzer()

        # Performance tracking
        self.learning_efficiency_history = deque(maxlen=50)
        self.adaptation_success_rate = 0.0

        logger.info("🧠 Advanced Adaptive Learning System initialized")

    def _build_meta_learning_network(self) -> nn.Module:
        """Build meta-learning network for learning how to learn"""
        return nn.Sequential(
            nn.Linear(15, 32),  # System state + learning context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),   # Learning parameter adjustments
            nn.Tanh()
        )

    def _build_mistake_analyzer(self) -> nn.Module:
        """Build mistake analyzer for learning from errors"""
        return nn.Sequential(
            nn.Linear(10, 20),  # Error context
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5),   # Root cause analysis + correction strategy
            nn.Softmax(dim=0)
        )

    @property
    def current_phase(self) -> LearningPhase:
        """Get current learning phase"""
        return self.parameter_adaptor.current_phase

    @property
    def phase_duration(self) -> int:
        """Get current phase duration"""
        return self.parameter_adaptor.phase_duration

    async def assess_learning_performance(self) -> LearningMetrics:
        """Comprehensive learning performance assessment"""

        metrics = await self.performance_assessor.assess_learning_performance(
            self.learning_history,
            self.mistake_database,
            self.creative_solutions,
            self.current_phase
        )

        self.learning_history.append(metrics)
        self.learning_efficiency_history.append(metrics.overall_learning_efficiency)

        return metrics

    async def adapt_learning_parameters(self) -> Dict[str, Any]:
        """Dynamically adapt learning parameters based on current performance"""

        logger.info("🔄 Adapting learning parameters based on performance")

        # Assess current learning performance
        current_metrics = await self.assess_learning_performance()

        # Delegate to parameter adaptor
        adaptation_results = await self.parameter_adaptor.adapt_learning_parameters(current_metrics)

        # Sync learning rate with adaptor
        self.current_learning_rate = self.parameter_adaptor.current_learning_rate

        # Update adaptation success rate from performance assessor
        self.adaptation_success_rate = current_metrics.parameter_adaptation_success

        logger.info(f"✅ Learning adaptation complete: {len(adaptation_results['adaptations_applied'])} changes applied")

        return adaptation_results

    async def learn_from_mistake(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced mistake learning with root cause analysis"""

        mistake_analysis = await self.mistake_learner.learn_from_mistake(
            error_context,
            self.mistake_database
        )

        # Sync learning rate with mistake learner
        self.current_learning_rate = self.mistake_learner.current_learning_rate

        return mistake_analysis

    async def generate_creative_solution(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate creative solutions using the creativity engine"""

        creative_solution = await self.creative_engine.generate_creative_solution(
            problem_context,
            self.creative_solutions
        )

        return creative_solution

    def get_learning_report(self) -> str:
        """Generate comprehensive learning system report"""

        if not self.learning_history:
            return "No learning data available"

        latest_metrics = self.learning_history[-1]

        report = []
        report.append("🧠 GARDEN OF CONSCIOUSNESS - ADAPTIVE LEARNING REPORT")
        report.append("=" * 70)
        report.append("")

        # Current Learning Status
        report.append(f"📚 CURRENT LEARNING STATUS:")
        report.append(f"   • Learning Phase: {latest_metrics.learning_phase.value.upper()}")
        report.append(f"   • Learning Rate: {self.current_learning_rate:.6f}")
        report.append(f"   • Phase Duration: {self.phase_duration} cycles")
        report.append(f"   • Overall Efficiency: {latest_metrics.overall_learning_efficiency:.3f}")
        report.append("")

        # Learning Performance Metrics
        report.append("📊 LEARNING PERFORMANCE METRICS:")
        report.append(f"   • Adaptation Rate: {latest_metrics.adaptation_rate:.3f}")
        report.append(f"   • Error Reduction Rate: {latest_metrics.error_reduction_rate:.3f}")
        report.append(f"   • Pattern Recognition: {latest_metrics.pattern_recognition_accuracy:.3f}")
        report.append(f"   • Creative Generation: {latest_metrics.creative_generation_score:.3f}")
        report.append(f"   • Mistake Learning: {latest_metrics.mistake_learning_effectiveness:.3f}")
        report.append(f"   • Parameter Adaptation: {latest_metrics.parameter_adaptation_success:.3f}")
        report.append("")

        # Learning Progress Analysis
        if len(self.learning_efficiency_history) > 5:
            recent_efficiency = list(self.learning_efficiency_history)[-10:]
            efficiency_trend = np.polyfit(range(len(recent_efficiency)), recent_efficiency, 1)[0]
            trend_symbol = "📈" if efficiency_trend > 0.01 else "📉" if efficiency_trend < -0.01 else "➡️"

            report.append("📈 LEARNING PROGRESS ANALYSIS:")
            report.append(f"   • Efficiency Trend: {trend_symbol} {efficiency_trend:+.4f} per cycle")
            report.append(f"   • Best Recent Efficiency: {max(recent_efficiency):.3f}")
            report.append(f"   • Learning Stability: {1.0 - np.var(recent_efficiency):.3f}")

        # Mistake Learning Analysis
        report.append("")
        report.append("🔍 MISTAKE LEARNING ANALYSIS:")
        report.append(f"   • Total Mistakes Analyzed: {len(self.mistake_database)}")

        if self.mistake_database:
            recent_mistakes = list(self.mistake_database)[-5:]
            avg_effectiveness = np.mean([m['learning_effectiveness'] for m in recent_mistakes])
            report.append(f"   • Recent Learning Effectiveness: {avg_effectiveness:.3f}")

            # Most common root causes
            root_causes = [m['root_cause_analysis']['most_likely_cause'] for m in recent_mistakes]
            if root_causes:
                most_common = max(set(root_causes), key=root_causes.count)
                report.append(f"   • Most Common Root Cause: {most_common}")

        # Creative Solutions
        report.append("")
        report.append("🎨 CREATIVE GENERATION ANALYSIS:")
        report.append(f"   • Total Creative Solutions: {len(self.creative_solutions)}")

        if self.creative_solutions:
            recent_solutions = list(self.creative_solutions)[-5:]
            avg_quality = np.mean([s['quality_score'] for s in recent_solutions])
            avg_novelty = np.mean([s['novelty_score'] for s in recent_solutions])
            report.append(f"   • Recent Solution Quality: {avg_quality:.3f}")
            report.append(f"   • Recent Solution Novelty: {avg_novelty:.3f}")

        # Recommendations
        report.append("")
        report.append("💡 LEARNING OPTIMIZATION RECOMMENDATIONS:")

        if latest_metrics.error_reduction_rate < 0.4:
            report.append("   🎯 Priority: Improve error reduction mechanisms")
        if latest_metrics.pattern_recognition_accuracy < 0.6:
            report.append("   🎯 Priority: Enhance pattern recognition capabilities")
        if latest_metrics.creative_generation_score < 0.5:
            report.append("   🎯 Priority: Boost creative solution generation")
        if latest_metrics.mistake_learning_effectiveness < 0.6:
            report.append("   🎯 Priority: Strengthen mistake learning processes")

        if latest_metrics.overall_learning_efficiency > 0.8:
            report.append("   ✅ Excellent learning performance - consider advancing to next phase")

        return "\n".join(report)


# Integration function
async def integrate_adaptive_learning(consciousness_system):
    """Integrate adaptive learning system with consciousness system"""

    logger.info("🧠 Integrating Advanced Adaptive Learning System")

    # Create learning system
    learning_system = AdaptiveLearningSystem(consciousness_system)

    # Initial assessment
    initial_metrics = await learning_system.assess_learning_performance()
    logger.info(f"Initial learning efficiency: {initial_metrics.overall_learning_efficiency:.3f}")

    # Apply initial adaptations
    adaptation_results = await learning_system.adapt_learning_parameters()
    logger.info(f"Initial adaptations applied: {len(adaptation_results['adaptations_applied'])}")

    # Generate and display report
    report = learning_system.get_learning_report()
    print("\n" + report)

    return learning_system


__all__ = ['AdaptiveLearningSystem', 'integrate_adaptive_learning']
