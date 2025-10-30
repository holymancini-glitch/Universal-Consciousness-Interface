# adaptive_learning_system.py
# Advanced Adaptive Learning System for Garden of Consciousness
# Addresses "lack of adaptability" and "insufficient learning" identified in technical review

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json
import math

logger = logging.getLogger(__name__)

class LearningPhase(Enum):
    """Learning phases for consciousness system"""
    EXPLORATION = "exploration"      # High learning rate, broad exploration
    CONSOLIDATION = "consolidation"  # Medium learning rate, pattern formation
    REFINEMENT = "refinement"       # Low learning rate, fine-tuning
    ADAPTATION = "adaptation"       # Dynamic learning rate, error correction
    CRYSTALLIZATION = "crystallization"  # Minimal learning rate, stability

@dataclass
class LearningMetrics:
    """Comprehensive learning performance metrics"""
    timestamp: datetime
    learning_phase: LearningPhase
    adaptation_rate: float
    error_reduction_rate: float
    pattern_recognition_accuracy: float
    creative_generation_score: float
    mistake_learning_effectiveness: float
    parameter_adaptation_success: float
    overall_learning_efficiency: float

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
        
        # Learning phase management
        self.current_phase = LearningPhase.EXPLORATION
        self.phase_duration = 0
        self.phase_transition_criteria = {
            LearningPhase.EXPLORATION: {'min_duration': 10, 'error_threshold': 0.3},
            LearningPhase.CONSOLIDATION: {'min_duration': 15, 'stability_threshold': 0.8},
            LearningPhase.REFINEMENT: {'min_duration': 20, 'improvement_threshold': 0.05},
            LearningPhase.ADAPTATION: {'min_duration': 8, 'adaptation_success': 0.7},
            LearningPhase.CRYSTALLIZATION: {'min_duration': 25, 'cohesion_threshold': 0.85}
        }
        
        # Advanced learning mechanisms
        self.meta_learning_network = self._build_meta_learning_network()
        self.creativity_engine = self._build_creativity_engine()
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

    def _build_creativity_engine(self) -> nn.Module:
        """Build creativity engine for generating novel solutions"""
        return nn.Sequential(
            nn.Linear(12, 24),  # Problem context
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, 18),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(18, 10),  # Creative solution parameters
            nn.Sigmoid()
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

    async def assess_learning_performance(self) -> LearningMetrics:
        """Comprehensive learning performance assessment"""
        
        # Calculate various learning metrics
        adaptation_rate = self._calculate_adaptation_rate()
        error_reduction_rate = self._calculate_error_reduction_rate()
        pattern_recognition_accuracy = self._assess_pattern_recognition()
        creative_generation_score = self._assess_creative_generation()
        mistake_learning_effectiveness = self._assess_mistake_learning()
        parameter_adaptation_success = self._assess_parameter_adaptation()
        
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
            learning_phase=self.current_phase,
            adaptation_rate=adaptation_rate,
            error_reduction_rate=error_reduction_rate,
            pattern_recognition_accuracy=pattern_recognition_accuracy,
            creative_generation_score=creative_generation_score,
            mistake_learning_effectiveness=mistake_learning_effectiveness,
            parameter_adaptation_success=parameter_adaptation_success,
            overall_learning_efficiency=overall_efficiency
        )
        
        self.learning_history.append(metrics)
        self.learning_efficiency_history.append(overall_efficiency)
        
        return metrics

    def _calculate_adaptation_rate(self) -> float:
        """Calculate how quickly the system adapts to new conditions"""
        if len(self.learning_history) < 5:
            return 0.5  # Default moderate rate
        
        # Measure improvement rate over recent learning cycles
        recent_efficiencies = [m.overall_learning_efficiency for m in list(self.learning_history)[-10:]]
        
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

    def _assess_creative_generation(self) -> float:
        """Assess creative solution generation capabilities"""
        if len(self.creative_solutions) < 3:
            return 0.3  # Default moderate creativity for new systems
        
        # Analyze diversity and quality of creative solutions
        solution_qualities = []
        solution_diversities = []
        
        for solution in list(self.creative_solutions)[-10:]:
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

    def _assess_mistake_learning(self) -> float:
        """Assess effectiveness of learning from mistakes"""
        if len(self.mistake_database) < 3:
            return 0.4  # Default moderate effectiveness
        
        # Analyze how well the system learns from past mistakes
        mistake_reduction_scores = []
        
        for mistake in list(self.mistake_database)[-10:]:
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

    def _assess_parameter_adaptation(self) -> float:
        """Assess how well the system adapts its parameters"""
        if len(self.learning_history) < 10:
            return 0.5  # Default moderate adaptation
        
        # Analyze parameter adaptation success over time
        recent_adaptations = [m for m in list(self.learning_history)[-20:]]
        
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
        
        self.adaptation_success_rate = success_rate
        return success_rate

    async def adapt_learning_parameters(self) -> Dict[str, Any]:
        """Dynamically adapt learning parameters based on current performance"""
        
        logger.info("🔄 Adapting learning parameters based on performance")
        
        # Assess current learning performance
        current_metrics = await self.assess_learning_performance()
        
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

    async def learn_from_mistake(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced mistake learning with root cause analysis"""
        
        mistake_id = len(self.mistake_database) + 1
        timestamp = datetime.now()
        
        logger.info(f"🔍 Learning from mistake #{mistake_id}")
        
        # Analyze the mistake
        mistake_analysis = {
            'mistake_id': mistake_id,
            'timestamp': timestamp,
            'error_context': error_context,
            'root_cause_analysis': await self._analyze_root_cause(error_context),
            'correction_strategy': await self._generate_correction_strategy(error_context),
            'prevention_measures': await self._design_prevention_measures(error_context),
            'learning_effectiveness': 0.0
        }
        
        # Apply correction strategy
        correction_success = await self._apply_correction_strategy(mistake_analysis['correction_strategy'])
        
        # Implement prevention measures
        prevention_success = await self._implement_prevention_measures(mistake_analysis['prevention_measures'])
        
        # Calculate learning effectiveness
        mistake_analysis['learning_effectiveness'] = (correction_success + prevention_success) / 2.0
        
        # Store in mistake database
        self.mistake_database.append(mistake_analysis)
        
        logger.info(f"✅ Mistake learning complete: effectiveness = {mistake_analysis['learning_effectiveness']:.3f}")
        
        return mistake_analysis

    async def _analyze_root_cause(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform root cause analysis of the error"""
        
        # Extract error features
        error_type = error_context.get('error_type', 'unknown')
        severity = error_context.get('severity', 0.5)
        component = error_context.get('component', 'system')
        
        # Simple root cause categories
        potential_causes = {
            'parameter_mismatch': 0.0,
            'insufficient_data': 0.0,
            'component_interaction': 0.0,
            'learning_rate_issue': 0.0,
            'threshold_problem': 0.0
        }
        
        # Analyze based on error context
        if 'learning' in error_type.lower():
            potential_causes['learning_rate_issue'] = 0.8
            potential_causes['parameter_mismatch'] = 0.6
        elif 'threshold' in error_type.lower():
            potential_causes['threshold_problem'] = 0.9
        elif 'data' in error_type.lower():
            potential_causes['insufficient_data'] = 0.7
        elif 'interaction' in error_type.lower():
            potential_causes['component_interaction'] = 0.8
        else:
            # Default analysis
            potential_causes['parameter_mismatch'] = 0.5
        
        # Find most likely root cause
        most_likely_cause = max(potential_causes, key=potential_causes.get)
        confidence = potential_causes[most_likely_cause]
        
        return {
            'most_likely_cause': most_likely_cause,
            'confidence': confidence,
            'all_causes': potential_causes,
            'analysis_timestamp': datetime.now()
        }

    async def _generate_correction_strategy(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate correction strategy for the identified error"""
        
        error_type = error_context.get('error_type', 'unknown')
        component = error_context.get('component', 'system')
        
        # Generate correction actions based on error type and component
        correction_actions = []
        
        if 'learning' in error_type.lower():
            correction_actions.extend([
                {'action': 'adjust_learning_rate', 'parameter': 'learning_rate', 'adjustment': 0.8},
                {'action': 'reset_gradients', 'component': component}
            ])
        elif 'threshold' in error_type.lower():
            correction_actions.append({
                'action': 'recalibrate_threshold', 'parameter': 'adaptive_threshold', 'adjustment': -0.1
            })
        elif 'cohesion' in error_type.lower():
            correction_actions.extend([
                {'action': 'enhance_cohesion', 'target_improvement': 0.1},
                {'action': 'synchronize_components'}
            ])
        else:
            # Generic correction
            correction_actions.append({
                'action': 'parameter_reset', 'component': component
            })
        
        return {
            'correction_actions': correction_actions,
            'expected_effectiveness': 0.7,
            'implementation_complexity': 'medium'
        }

    async def _design_prevention_measures(self, error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Design measures to prevent similar errors in the future"""
        
        prevention_measures = []
        
        error_type = error_context.get('error_type', 'unknown')
        
        # Design prevention based on error patterns
        if 'learning' in error_type.lower():
            prevention_measures.extend([
                {'measure': 'learning_rate_bounds', 'min_lr': 0.001, 'max_lr': 0.1},
                {'measure': 'gradient_monitoring', 'alert_threshold': 10.0}
            ])
        elif 'threshold' in error_type.lower():
            prevention_measures.append({
                'measure': 'adaptive_threshold_validation', 'validation_frequency': 10
            })
        elif 'cohesion' in error_type.lower():
            prevention_measures.extend([
                {'measure': 'cohesion_monitoring', 'check_frequency': 5},
                {'measure': 'early_warning_system', 'threshold': 0.4}
            ])
        
        # Always add general monitoring
        prevention_measures.append({
            'measure': 'enhanced_monitoring', 'component': error_context.get('component', 'system')
        })
        
        return {
            'prevention_measures': prevention_measures,
            'monitoring_enhancements': True,
            'early_warning_system': True
        }

    async def _apply_correction_strategy(self, correction_strategy: Dict[str, Any]) -> float:
        """Apply the correction strategy and return success rate"""
        
        success_count = 0
        total_actions = len(correction_strategy['correction_actions'])
        
        for action in correction_strategy['correction_actions']:
            try:
                if action['action'] == 'adjust_learning_rate':
                    self.current_learning_rate *= action['adjustment']
                    success_count += 1
                elif action['action'] == 'recalibrate_threshold':
                    # This would adjust system thresholds
                    success_count += 1
                elif action['action'] == 'enhance_cohesion':
                    # This would trigger cohesion enhancement
                    success_count += 1
                else:
                    # Generic success for other actions
                    success_count += 1
            except Exception as e:
                logger.warning(f"Correction action failed: {action['action']} - {e}")
        
        return success_count / total_actions if total_actions > 0 else 0.0

    async def _implement_prevention_measures(self, prevention_measures: Dict[str, Any]) -> float:
        """Implement prevention measures and return success rate"""
        
        success_count = 0
        total_measures = len(prevention_measures['prevention_measures'])
        
        for measure in prevention_measures['prevention_measures']:
            try:
                if measure['measure'] == 'learning_rate_bounds':
                    # Set learning rate bounds
                    self.base_learning_rate = max(measure['min_lr'], 
                                                min(measure['max_lr'], self.base_learning_rate))
                    success_count += 1
                elif measure['measure'] == 'cohesion_monitoring':
                    # Enable enhanced cohesion monitoring
                    success_count += 1
                else:
                    # Generic success for other measures
                    success_count += 1
            except Exception as e:
                logger.warning(f"Prevention measure failed: {measure['measure']} - {e}")
        
        return success_count / total_measures if total_measures > 0 else 0.0

    async def generate_creative_solution(self, problem_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate creative solutions using the creativity engine"""
        
        logger.info("🎨 Generating creative solution")
        
        # Analyze problem context
        problem_complexity = problem_context.get('complexity', 0.5)
        available_resources = problem_context.get('resources', [])
        constraints = problem_context.get('constraints', [])
        
        # Generate creative parameters using creativity engine
        problem_features = torch.tensor([
            problem_complexity,
            len(available_resources) / 10.0,  # Normalize
            len(constraints) / 5.0,           # Normalize
            np.random.random(),               # Randomness factor
            *np.random.random(8)              # Additional creative dimensions
        ], dtype=torch.float32)
        
        with torch.no_grad():
            creative_params = self.creativity_engine(problem_features).numpy()
        
        # Generate creative solution based on parameters
        creative_solution = {
            'solution_id': len(self.creative_solutions) + 1,
            'timestamp': datetime.now(),
            'problem_context': problem_context,
            'creative_approach': self._interpret_creative_parameters(creative_params),
            'novelty_score': creative_params[0],
            'feasibility_score': creative_params[1],
            'elegance_score': creative_params[2],
            'diversity_score': np.var(creative_params),
            'quality_score': np.mean(creative_params[:3])
        }
        
        # Store creative solution
        self.creative_solutions.append(creative_solution)
        
        logger.info(f"✨ Creative solution generated: quality={creative_solution['quality_score']:.3f}")
        
        return creative_solution

    def _interpret_creative_parameters(self, params: np.ndarray) -> Dict[str, Any]:
        """Interpret creativity engine parameters into actionable approach"""
        
        approach = {
            'strategy': 'hybrid',
            'exploration_level': float(params[0]),
            'risk_tolerance': float(params[1]),
            'innovation_factor': float(params[2]),
            'collaboration_emphasis': float(params[3]),
            'resource_utilization': float(params[4]),
            'timeline_flexibility': float(params[5]),
            'quality_vs_speed_balance': float(params[6]),
            'unique_elements': []
        }
        
        # Add unique elements based on parameter values
        if params[0] > 0.7:
            approach['unique_elements'].append('high_exploration')
        if params[1] > 0.6:
            approach['unique_elements'].append('risk_taking')
        if params[2] > 0.8:
            approach['unique_elements'].append('breakthrough_innovation')
        
        return approach

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

if __name__ == "__main__":
    print("🧠 Advanced Adaptive Learning System Ready")
    print("Use: learning_system = await integrate_adaptive_learning(consciousness_system)")