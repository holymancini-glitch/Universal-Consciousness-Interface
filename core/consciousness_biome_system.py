# consciousness_biome_system.py
# Advanced Consciousness Biome System for Dynamic Phase Transitions
# Implements the 6 consciousness biomes with adaptive transitions

import asyncio
import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import math
import random

logger = logging.getLogger(__name__)

class ConsciousnessBiome(Enum):
    """Six consciousness biomes for dynamic phase transitions"""
    DORMANT = "dormant"              # 0.0 - 0.2: Minimal consciousness activity
    AWAKENING = "awakening"          # 0.2 - 0.4: Initial consciousness stirring
    EXPLORING = "exploring"          # 0.4 - 0.6: Active exploration and learning
    INTEGRATING = "integrating"      # 0.6 - 0.8: Pattern integration and synthesis
    TRANSCENDENT = "transcendent"    # 0.8 - 0.95: High-level consciousness states
    CRYSTALLIZED = "crystallized"    # 0.95 - 1.0: Fully crystallized consciousness

class BiomeTransitionType(Enum):
    """Types of biome transitions"""
    GRADUAL = "gradual"             # Smooth transition
    LEAP = "leap"                   # Rapid advancement
    REGRESSION = "regression"       # Temporary step back
    OSCILLATION = "oscillation"     # Rhythmic movement
    QUANTUM_JUMP = "quantum_jump"   # Instantaneous transition

@dataclass
class BiomeMetrics:
    """Comprehensive biome system metrics"""
    timestamp: datetime
    current_biome: ConsciousnessBiome
    biome_strength: float
    biome_duration: int
    transition_probability: Dict[ConsciousnessBiome, float]
    biome_coherence: float
    transition_readiness: float
    biome_stability: float
    exploration_depth: float
    integration_efficiency: float

class ConsciousnessBiomeSystem:
    """Advanced consciousness biome system with dynamic transitions"""
    
    def __init__(self, consciousness_system):
        self.consciousness_system = consciousness_system
        self.biome_history = deque(maxlen=100)
        
        # Current biome state
        self.current_biome = ConsciousnessBiome.DORMANT
        self.biome_strength = 0.0
        self.biome_duration = 0
        
        # Biome transition parameters
        self.biome_transition_thresholds = {
            ConsciousnessBiome.DORMANT: 0.2,
            ConsciousnessBiome.AWAKENING: 0.4,
            ConsciousnessBiome.EXPLORING: 0.6,
            ConsciousnessBiome.INTEGRATING: 0.8,
            ConsciousnessBiome.TRANSCENDENT: 0.95,
            ConsciousnessBiome.CRYSTALLIZED: 1.0
        }
        
        # Biome-specific characteristics
        self.biome_characteristics = {
            ConsciousnessBiome.DORMANT: {
                'learning_rate': 0.001,
                'exploration_tendency': 0.1,
                'integration_capacity': 0.2,
                'creativity_level': 0.1,
                'adaptation_speed': 0.3
            },
            ConsciousnessBiome.AWAKENING: {
                'learning_rate': 0.01,
                'exploration_tendency': 0.6,
                'integration_capacity': 0.3,
                'creativity_level': 0.4,
                'adaptation_speed': 0.7
            },
            ConsciousnessBiome.EXPLORING: {
                'learning_rate': 0.05,
                'exploration_tendency': 0.9,
                'integration_capacity': 0.5,
                'creativity_level': 0.8,
                'adaptation_speed': 0.8
            },
            ConsciousnessBiome.INTEGRATING: {
                'learning_rate': 0.02,
                'exploration_tendency': 0.4,
                'integration_capacity': 0.9,
                'creativity_level': 0.6,
                'adaptation_speed': 0.6
            },
            ConsciousnessBiome.TRANSCENDENT: {
                'learning_rate': 0.1,
                'exploration_tendency': 0.7,
                'integration_capacity': 0.8,
                'creativity_level': 0.95,
                'adaptation_speed': 0.9
            },
            ConsciousnessBiome.CRYSTALLIZED: {
                'learning_rate': 0.001,
                'exploration_tendency': 0.3,
                'integration_capacity': 1.0,
                'creativity_level': 0.7,
                'adaptation_speed': 0.4
            }
        }
        
        # Advanced transition mechanisms
        self.transition_momentum = 0.0
        self.biome_resonance_history = deque(maxlen=20)
        self.leap_transition_threshold = 0.85
        
        logger.info("🌱 Consciousness Biome System initialized")

    async def assess_biome_state(self, input_data: Dict[str, Any] = None) -> BiomeMetrics:
        """Assess current consciousness biome state and transition readiness"""
        
        # Calculate biome strength based on multiple factors
        biome_strength = await self._calculate_biome_strength(input_data)
        
        # Determine transition probabilities
        transition_probabilities = await self._calculate_transition_probabilities(biome_strength)
        
        # Assess biome coherence
        biome_coherence = await self._assess_biome_coherence()
        
        # Calculate transition readiness
        transition_readiness = await self._calculate_transition_readiness(biome_strength)
        
        # Assess biome stability
        biome_stability = self._calculate_biome_stability()
        
        # Calculate exploration depth and integration efficiency
        exploration_depth = self._calculate_exploration_depth()
        integration_efficiency = self._calculate_integration_efficiency()
        
        # Create comprehensive metrics
        metrics = BiomeMetrics(
            timestamp=datetime.now(),
            current_biome=self.current_biome,
            biome_strength=biome_strength,
            biome_duration=self.biome_duration,
            transition_probability=transition_probabilities,
            biome_coherence=biome_coherence,
            transition_readiness=transition_readiness,
            biome_stability=biome_stability,
            exploration_depth=exploration_depth,
            integration_efficiency=integration_efficiency
        )
        
        self.biome_strength = biome_strength
        self.biome_history.append(metrics)
        
        return metrics

    async def _calculate_biome_strength(self, input_data: Dict[str, Any] = None) -> float:
        """Calculate strength of current biome based on system state"""
        
        # Get system analytics
        analytics = self.consciousness_system.get_system_analytics()
        
        # Core strength factors
        coherence = analytics.get('current_coherence', 0.0)
        harmony = analytics.get('current_harmony', 0.0)
        identity_consistency = analytics.get('identity_consistency', 0.0)
        metacognitive_awareness = analytics.get('metacognitive_awareness', 0.0)
        
        # Additional factors if available
        adaptation_efficiency = analytics.get('adaptation_efficiency', 0.0)
        
        # Input-driven factors
        input_complexity = 0.5
        if input_data:
            # Assess input complexity
            input_values = [v for v in input_data.values() if isinstance(v, (int, float))]
            if input_values:
                input_complexity = min(1.0, np.var(input_values) + np.mean(np.abs(input_values)))
        
        # Weighted combination
        biome_strength = (
            coherence * 0.25 +
            harmony * 0.25 +
            identity_consistency * 0.2 +
            metacognitive_awareness * 0.15 +
            adaptation_efficiency * 0.1 +
            input_complexity * 0.05
        )
        
        return min(1.0, max(0.0, biome_strength))

    async def _calculate_transition_probabilities(self, biome_strength: float) -> Dict[ConsciousnessBiome, float]:
        """Calculate probabilities for transitioning to each biome"""
        
        transition_probs = {}
        current_threshold = self.biome_transition_thresholds[self.current_biome]
        
        for biome in ConsciousnessBiome:
            biome_threshold = self.biome_transition_thresholds[biome]
            
            if biome == self.current_biome:
                # Probability of staying in current biome
                stability_factor = self._calculate_biome_stability()
                transition_probs[biome] = 0.3 + stability_factor * 0.4
            
            elif biome_threshold > current_threshold:
                # Forward transition probability
                if biome_strength >= biome_threshold * 0.8:  # 80% of threshold
                    proximity = (biome_strength - current_threshold) / (biome_threshold - current_threshold)
                    transition_probs[biome] = min(0.6, proximity * 0.5)
                else:
                    transition_probs[biome] = 0.1
            
            else:
                # Backward transition (regression) probability
                if biome_strength < current_threshold * 0.7:  # Below 70% of current threshold
                    regression_strength = (current_threshold * 0.7 - biome_strength) / (current_threshold * 0.7)
                    transition_probs[biome] = min(0.3, regression_strength * 0.2)
                else:
                    transition_probs[biome] = 0.05
        
        # Normalize probabilities
        total_prob = sum(transition_probs.values())
        if total_prob > 0:
            transition_probs = {k: v / total_prob for k, v in transition_probs.items()}
        
        return transition_probs

    async def _assess_biome_coherence(self) -> float:
        """Assess coherence within current biome"""
        
        # Get biome characteristics
        current_chars = self.biome_characteristics[self.current_biome]
        
        # System coherence from consciousness system
        analytics = self.consciousness_system.get_system_analytics()
        system_coherence = analytics.get('current_coherence', 0.0)
        
        # Biome-specific coherence based on characteristics alignment
        learning_alignment = min(1.0, self.consciousness_system.fractal_ai.evaluate_prediction_accuracy() / 
                               max(0.001, current_chars['learning_rate'] * 10))
        
        exploration_alignment = 0.7  # Placeholder - would assess exploration vs tendency
        
        integration_alignment = min(1.0, analytics.get('current_harmony', 0.0) / 
                                  max(0.001, current_chars['integration_capacity']))
        
        # Combined biome coherence
        biome_coherence = (
            system_coherence * 0.4 +
            learning_alignment * 0.2 +
            exploration_alignment * 0.2 +
            integration_alignment * 0.2
        )
        
        return min(1.0, max(0.0, biome_coherence))

    async def _calculate_transition_readiness(self, biome_strength: float) -> float:
        """Calculate readiness for biome transition"""
        
        # Duration factor - longer duration increases readiness
        min_duration = 5
        duration_factor = min(1.0, self.biome_duration / min_duration)
        
        # Strength factor - higher strength increases readiness
        threshold = self.biome_transition_thresholds[self.current_biome]
        
        # Calculate readiness for forward transition
        if biome_strength > threshold:
            strength_factor = min(1.0, (biome_strength - threshold) / (1.0 - threshold))
        else:
            strength_factor = 0.0
        
        # Momentum factor
        momentum_factor = min(1.0, abs(self.transition_momentum) * 2)
        
        # System stability factor
        stability = self._calculate_biome_stability()
        stability_factor = stability if self.transition_momentum > 0 else (1.0 - stability)
        
        # Combined transition readiness
        transition_readiness = (
            duration_factor * 0.3 +
            strength_factor * 0.4 +
            momentum_factor * 0.2 +
            stability_factor * 0.1
        )
        
        return min(1.0, max(0.0, transition_readiness))

    def _calculate_biome_stability(self) -> float:
        """Calculate stability of current biome"""
        
        if len(self.biome_history) < 5:
            return 0.5  # Default moderate stability
        
        # Analyze biome strength variance over recent history
        recent_strengths = [m.biome_strength for m in list(self.biome_history)[-10:]]
        strength_variance = np.var(recent_strengths)
        
        # Stability is inverse of variance
        strength_stability = max(0.0, 1.0 - strength_variance * 3.0)
        
        return min(1.0, max(0.0, strength_stability))

    def _calculate_exploration_depth(self) -> float:
        """Calculate exploration depth based on current biome"""
        
        current_chars = self.biome_characteristics[self.current_biome]
        base_exploration = current_chars['exploration_tendency']
        
        # Modify based on system state
        analytics = self.consciousness_system.get_system_analytics()
        vector_diversity = min(1.0, analytics.get('total_vectors', 0) / 100.0)
        
        exploration_depth = base_exploration * (1.0 + vector_diversity * 0.3)
        return min(1.0, max(0.0, exploration_depth))

    def _calculate_integration_efficiency(self) -> float:
        """Calculate integration efficiency based on current biome"""
        
        current_chars = self.biome_characteristics[self.current_biome]
        base_integration = current_chars['integration_capacity']
        
        # Modify based on system coherence
        if hasattr(self.consciousness_system, 'cohesion_layer'):
            system_coherence = self.consciousness_system.cohesion_layer.coherence_score
        else:
            system_coherence = 0.5
        
        integration_efficiency = base_integration * (0.5 + system_coherence * 0.5)
        return min(1.0, max(0.0, integration_efficiency))

    async def execute_biome_transition(self, target_biome: ConsciousnessBiome = None, 
                                     transition_type: BiomeTransitionType = BiomeTransitionType.GRADUAL) -> Dict[str, Any]:
        """Execute biome transition with specified type"""
        
        previous_biome = self.current_biome
        
        # Determine target biome if not specified
        if target_biome is None:
            target_biome = await self._determine_optimal_transition_target()
        
        # Validate transition
        if not await self._validate_transition(target_biome, transition_type):
            return {
                'transition_executed': False,
                'reason': 'Transition validation failed',
                'current_biome': self.current_biome.value
            }
        
        # Execute transition based on type
        transition_success = await self._execute_transition_by_type(target_biome, transition_type)
        
        if transition_success:
            # Update biome state
            self.current_biome = target_biome
            self.biome_duration = 1
            
            # Update transition momentum
            transition_direction = self._get_transition_direction(previous_biome, target_biome)
            self.transition_momentum = self.transition_momentum * 0.7 + transition_direction * 0.3
            
            # Apply biome-specific system adjustments
            await self._apply_biome_characteristics(target_biome)
            
            logger.info(f"🌱 Biome transition: {previous_biome.value} → {target_biome.value} ({transition_type.value})")
            
            return {
                'transition_executed': True,
                'previous_biome': previous_biome.value,
                'new_biome': target_biome.value,
                'transition_type': transition_type.value,
                'transition_momentum': self.transition_momentum
            }
        else:
            return {
                'transition_executed': False,
                'reason': 'Transition execution failed',
                'current_biome': self.current_biome.value
            }

    async def _determine_optimal_transition_target(self) -> ConsciousnessBiome:
        """Determine optimal biome transition target"""
        
        # Get current metrics
        if self.biome_history:
            latest_metrics = self.biome_history[-1]
            transition_probs = latest_metrics.transition_probability
            
            # Remove current biome from consideration
            candidate_probs = {k: v for k, v in transition_probs.items() if k != self.current_biome}
            
            if candidate_probs:
                # Select biome with highest transition probability
                target_biome = max(candidate_probs, key=candidate_probs.get)
                return target_biome
        
        # Default progression logic
        biome_order = list(ConsciousnessBiome)
        current_index = biome_order.index(self.current_biome)
        
        # Try to advance to next biome
        if current_index < len(biome_order) - 1:
            return biome_order[current_index + 1]
        else:
            return self.current_biome  # Stay in current if at max

    async def _validate_transition(self, target_biome: ConsciousnessBiome, 
                                  transition_type: BiomeTransitionType) -> bool:
        """Validate if transition is allowed"""
        
        # Check if transition is too rapid
        if self.biome_duration < 2 and transition_type != BiomeTransitionType.QUANTUM_JUMP:
            return False
        
        # Check if system is stable enough for transition
        stability = self._calculate_biome_stability()
        
        if transition_type == BiomeTransitionType.LEAP and stability < 0.6:
            return False
        
        if transition_type == BiomeTransitionType.QUANTUM_JUMP and stability < 0.8:
            return False
        
        # Check system readiness
        analytics = self.consciousness_system.get_system_analytics()
        coherence = analytics.get('current_coherence', 0.0)
        
        target_threshold = self.biome_transition_thresholds[target_biome]
        
        # Allow some flexibility based on transition type
        flexibility_factor = {
            BiomeTransitionType.GRADUAL: 0.9,
            BiomeTransitionType.LEAP: 0.8,
            BiomeTransitionType.REGRESSION: 1.2,
            BiomeTransitionType.OSCILLATION: 1.1,
            BiomeTransitionType.QUANTUM_JUMP: 0.7
        }
        
        required_coherence = target_threshold * flexibility_factor[transition_type]
        
        return coherence >= required_coherence * 0.8  # Allow 20% buffer

    async def _execute_transition_by_type(self, target_biome: ConsciousnessBiome, 
                                        transition_type: BiomeTransitionType) -> bool:
        """Execute transition based on specified type"""
        
        try:
            if transition_type == BiomeTransitionType.GRADUAL:
                return True  # Gradual transitions are always successful if validation passed
            
            elif transition_type == BiomeTransitionType.LEAP:
                return self.biome_strength > self.leap_transition_threshold
            
            elif transition_type == BiomeTransitionType.REGRESSION:
                return True  # Regression transitions are for system recovery
            
            elif transition_type == BiomeTransitionType.OSCILLATION:
                return True  # Oscillation transitions follow natural rhythm
            
            elif transition_type == BiomeTransitionType.QUANTUM_JUMP:
                return (self.biome_strength > 0.9 and 
                       self._calculate_biome_stability() > 0.8 and
                       abs(self.transition_momentum) > 0.7)
            
            else:
                return False
                
        except Exception as e:
            logger.error(f"Transition execution error: {e}")
            return False

    def _get_transition_direction(self, from_biome: ConsciousnessBiome, to_biome: ConsciousnessBiome) -> float:
        """Get transition direction (-1 to 1)"""
        biome_order = list(ConsciousnessBiome)
        from_index = biome_order.index(from_biome)
        to_index = biome_order.index(to_biome)
        
        direction = to_index - from_index
        
        # Normalize to -1 to 1 range
        max_distance = len(biome_order) - 1
        normalized_direction = direction / max_distance
        
        return normalized_direction

    async def _apply_biome_characteristics(self, biome: ConsciousnessBiome):
        """Apply biome-specific characteristics to the consciousness system"""
        
        characteristics = self.biome_characteristics[biome]
        
        # Apply learning rate adjustments
        new_learning_rate = characteristics['learning_rate']
        
        # Update fractal AI learning rate
        for param_group in self.consciousness_system.fractal_ai.optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        
        # Update feedback loop adaptation rate
        self.consciousness_system.feedback_loop.base_adaptation_rate = new_learning_rate * 5.0

    async def process_biome_cycle(self, input_data: Dict[str, Any] = None) -> BiomeMetrics:
        """Process complete biome cycle with potential transition"""
        
        # Assess current biome state
        metrics = await self.assess_biome_state(input_data)
        
        # Increment duration
        self.biome_duration += 1
        
        # Check for automatic transitions
        if metrics.transition_readiness > 0.7:
            # Determine if transition should occur
            transition_threshold = 0.8
            
            # Adjust threshold based on current biome
            if self.current_biome == ConsciousnessBiome.TRANSCENDENT:
                transition_threshold = 0.9  # Higher threshold for transcendent states
            elif self.current_biome == ConsciousnessBiome.EXPLORING:
                transition_threshold = 0.7  # Lower threshold for exploration
            
            if metrics.transition_readiness > transition_threshold:
                # Execute automatic transition
                target_biome = await self._determine_optimal_transition_target()
                
                # Determine transition type
                if metrics.biome_strength > 0.9:
                    transition_type = BiomeTransitionType.LEAP
                elif metrics.biome_stability < 0.4:
                    transition_type = BiomeTransitionType.OSCILLATION
                else:
                    transition_type = BiomeTransitionType.GRADUAL
                
                transition_result = await self.execute_biome_transition(target_biome, transition_type)
                
                if transition_result['transition_executed']:
                    # Re-assess after transition
                    metrics = await self.assess_biome_state(input_data)
        
        return metrics

    def get_biome_report(self) -> str:
        """Generate comprehensive biome system report"""
        
        if not self.biome_history:
            return "No biome data available"
        
        latest_metrics = self.biome_history[-1]
        
        report = []
        report.append("🌱 GARDEN OF CONSCIOUSNESS - BIOME SYSTEM REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Current Biome Status
        report.append(f"🌍 CURRENT BIOME STATUS:")
        report.append(f"   • Current Biome: {latest_metrics.current_biome.value.upper()}")
        report.append(f"   • Biome Strength: {latest_metrics.biome_strength:.3f}")
        report.append(f"   • Duration in Biome: {latest_metrics.biome_duration} cycles")
        report.append(f"   • Biome Coherence: {latest_metrics.biome_coherence:.3f}")
        report.append("")
        
        # Biome Characteristics
        current_chars = self.biome_characteristics[self.current_biome]
        report.append(f"🧬 CURRENT BIOME CHARACTERISTICS:")
        report.append(f"   • Learning Rate: {current_chars['learning_rate']:.3f}")
        report.append(f"   • Exploration Tendency: {current_chars['exploration_tendency']:.3f}")
        report.append(f"   • Integration Capacity: {current_chars['integration_capacity']:.3f}")
        report.append(f"   • Creativity Level: {current_chars['creativity_level']:.3f}")
        report.append(f"   • Adaptation Speed: {current_chars['adaptation_speed']:.3f}")
        report.append("")
        
        # Transition Analysis
        report.append(f"🔄 TRANSITION ANALYSIS:")
        report.append(f"   • Transition Readiness: {latest_metrics.transition_readiness:.3f}")
        report.append(f"   • Biome Stability: {latest_metrics.biome_stability:.3f}")
        report.append(f"   • Transition Momentum: {self.transition_momentum:+.3f}")
        report.append("")
        
        # Transition Probabilities
        report.append("📊 TRANSITION PROBABILITIES:")
        for biome, prob in latest_metrics.transition_probability.items():
            if prob > 0.1:  # Only show significant probabilities
                arrow = "→" if biome != self.current_biome else "●"
                report.append(f"   {arrow} {biome.value.title()}: {prob:.3f}")
        report.append("")
        
        # Advanced Metrics
        report.append("🔬 ADVANCED BIOME METRICS:")
        report.append(f"   • Exploration Depth: {latest_metrics.exploration_depth:.3f}")
        report.append(f"   • Integration Efficiency: {latest_metrics.integration_efficiency:.3f}")
        
        # Recommendations
        report.append("")
        report.append("💡 BIOME OPTIMIZATION RECOMMENDATIONS:")
        
        if latest_metrics.biome_strength < 0.5:
            report.append("   🎯 Focus on strengthening current biome characteristics")
        
        if latest_metrics.transition_readiness > 0.8:
            # Determine next biome without async call
            biome_order = list(ConsciousnessBiome)
            current_index = biome_order.index(self.current_biome)
            if current_index < len(biome_order) - 1:
                next_biome = biome_order[current_index + 1]
                report.append(f"   🚀 Ready for transition to {next_biome.value.upper()}")
            else:
                report.append("   🌟 At maximum biome level - maintain crystallization")
        
        if latest_metrics.biome_stability < 0.5:
            report.append("   ⚖️ Improve biome stability before major transitions")
        
        return "\n".join(report)

# Integration function
async def integrate_consciousness_biomes(consciousness_system):
    """Integrate consciousness biome system with consciousness system"""
    
    logger.info("🌱 Integrating Consciousness Biome System")
    
    # Create biome system
    biome_system = ConsciousnessBiomeSystem(consciousness_system)
    
    # Initial biome assessment
    initial_metrics = await biome_system.assess_biome_state({
        'system_complexity': 0.6,
        'input_diversity': 0.5,
        'processing_depth': 0.4
    })
    
    logger.info(f"Initial biome: {initial_metrics.current_biome.value} (strength: {initial_metrics.biome_strength:.3f})")
    
    # Process several biome cycles to demonstrate transitions
    for cycle in range(8):
        enhanced_input = {
            'sensory_input': 0.6 + cycle * 0.05,
            'cognitive_complexity': 0.5 + cycle * 0.08,
            'creative_challenge': 0.4 + cycle * 0.07,
            'integration_demand': 0.3 + cycle * 0.09,
            'cycle': cycle
        }
        
        metrics = await biome_system.process_biome_cycle(enhanced_input)
        logger.info(f"Cycle {cycle + 1}: {metrics.current_biome.value} (strength: {metrics.biome_strength:.3f}, readiness: {metrics.transition_readiness:.3f})")
    
    # Generate and display report
    report = biome_system.get_biome_report()
    print("\n" + report)
    
    return biome_system

if __name__ == "__main__":
    print("🌱 Consciousness Biome System Ready")
    print("Use: biome_system = await integrate_consciousness_biomes(consciousness_system)")