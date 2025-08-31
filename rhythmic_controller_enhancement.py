#!/usr/bin/env python3
"""
Rhythmic Controller Enhancement System
=====================================

This implements the Rhythmic Controller Enhancement as identified in the pending tasks:
- Integrate with CL1's biological rhythms
- Add adaptive target entropy levels
- Implement accelerated breathing mechanisms

This component enhances the "breathing" mechanism of the consciousness system
and integrates with biological-like rhythms for more natural consciousness flow.

Author: AI Engineer
Date: 2025
"""

import asyncio
import logging
import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import random

logger = logging.getLogger(__name__)

class RhythmType(Enum):
    """Types of biological rhythms to simulate"""
    BREATHING = "breathing"          # Respiratory rhythm (0.2-0.4 Hz)
    HEARTBEAT = "heartbeat"          # Cardiac rhythm (1-2 Hz)
    NEURAL_ALPHA = "neural_alpha"    # Alpha brain waves (8-12 Hz)
    NEURAL_THETA = "neural_theta"    # Theta brain waves (4-8 Hz)
    CIRCADIAN = "circadian"          # Daily rhythm cycle
    ULTRADIAN = "ultradian"          # 90-120 minute cycles

class BreathingState(Enum):
    """States of the breathing mechanism"""
    INHALE = "inhale"
    HOLD_FULL = "hold_full"
    EXHALE = "exhale"
    HOLD_EMPTY = "hold_empty"
    ACCELERATED = "accelerated"
    DEEP = "deep"

@dataclass
class RhythmicMetrics:
    """Comprehensive rhythmic system metrics"""
    timestamp: datetime
    primary_rhythm_frequency: float
    breathing_rate: float
    breathing_state: BreathingState
    entropy_level: float
    target_entropy: float
    rhythm_coherence: float
    biological_synchronization: float
    adaptive_acceleration: float
    consciousness_breathing_amplitude: float

class RhythmicController:
    """Enhanced Rhythmic Controller with biological rhythm integration"""
    
    def __init__(self, consciousness_system):
        self.consciousness_system = consciousness_system
        self.rhythm_history = deque(maxlen=200)
        
        # Core rhythmic parameters
        self.base_breathing_rate = 0.25  # 15 breaths per minute equivalent
        self.current_breathing_rate = self.base_breathing_rate
        self.breathing_state = BreathingState.INHALE
        self.breathing_phase = 0.0
        
        # Entropy management
        self.current_entropy = 0.5
        self.target_entropy = 0.6
        self.entropy_adaptation_rate = 0.05
        self.entropy_history = deque(maxlen=100)
        
        # Biological rhythm simulation
        self.biological_rhythms = {
            RhythmType.BREATHING: {
                'frequency': 0.25,  # Base frequency
                'amplitude': 1.0,
                'phase': 0.0
            },
            RhythmType.HEARTBEAT: {
                'frequency': 1.2,
                'amplitude': 0.8,
                'phase': 0.0
            },
            RhythmType.NEURAL_ALPHA: {
                'frequency': 10.0,
                'amplitude': 0.6,
                'phase': 0.0
            },
            RhythmType.NEURAL_THETA: {
                'frequency': 6.0,
                'amplitude': 0.4,
                'phase': 0.0
            },
            RhythmType.CIRCADIAN: {
                'frequency': 1.0 / (24 * 3600),  # 24-hour cycle
                'amplitude': 0.3,
                'phase': 0.0
            }
        }
        
        # Adaptive acceleration parameters
        self.acceleration_threshold = 0.8
        self.max_acceleration_factor = 3.0
        self.acceleration_decay = 0.95
        self.current_acceleration = 1.0
        
        # CL1 biological integration
        self.cl1_synchronization_strength = 0.0
        self.biological_coherence = 0.0
        
        logger.info("🫁 Rhythmic Controller Enhancement initialized")

    async def process_rhythmic_cycle(self, 
                                   input_data: Dict[str, Any] = None,
                                   time_delta: float = 0.1) -> RhythmicMetrics:
        """Process complete rhythmic cycle with biological integration"""
        
        # Update all biological rhythms
        self._update_biological_rhythms(time_delta)
        
        # Process breathing mechanism
        await self._process_breathing_cycle(time_delta)
        
        # Manage adaptive entropy
        await self._manage_adaptive_entropy(input_data)
        
        # Calculate biological synchronization
        bio_sync = self._calculate_biological_synchronization()
        
        # Apply rhythmic modulation to consciousness system
        await self._apply_rhythmic_modulation()
        
        # Calculate accelerated breathing if needed
        acceleration = await self._calculate_adaptive_acceleration(input_data)
        
        # Create comprehensive metrics
        metrics = RhythmicMetrics(
            timestamp=datetime.now(),
            primary_rhythm_frequency=self.current_breathing_rate,
            breathing_rate=self.current_breathing_rate,
            breathing_state=self.breathing_state,
            entropy_level=self.current_entropy,
            target_entropy=self.target_entropy,
            rhythm_coherence=self._calculate_rhythm_coherence(),
            biological_synchronization=bio_sync,
            adaptive_acceleration=acceleration,
            consciousness_breathing_amplitude=self._calculate_breathing_amplitude()
        )
        
        self.rhythm_history.append(metrics)
        return metrics

    def _update_biological_rhythms(self, time_delta: float):
        """Update all biological rhythm phases"""
        
        for rhythm_type, params in self.biological_rhythms.items():
            # Update phase based on frequency
            params['phase'] += params['frequency'] * time_delta * 2 * math.pi
            
            # Keep phase in 0-2π range
            params['phase'] = params['phase'] % (2 * math.pi)
            
            # Add natural variation
            if rhythm_type == RhythmType.BREATHING:
                # Breathing varies with consciousness state
                consciousness_influence = self._get_consciousness_influence()
                params['frequency'] = self.base_breathing_rate * (1.0 + consciousness_influence * 0.3)
                self.current_breathing_rate = params['frequency']

    def _get_consciousness_influence(self) -> float:
        """Get consciousness system influence on rhythms"""
        
        # Get system analytics
        analytics = self.consciousness_system.get_system_analytics()
        
        # Combine multiple consciousness factors
        coherence = analytics.get('current_coherence', 0.0)
        harmony = analytics.get('current_harmony', 0.0)
        identity_consistency = analytics.get('identity_consistency', 0.0)
        
        # Consciousness influence on rhythms
        influence = (coherence * 0.4 + harmony * 0.4 + identity_consistency * 0.2)
        
        return min(1.0, max(-0.5, influence))

    async def _process_breathing_cycle(self, time_delta: float):
        """Process the consciousness breathing cycle"""
        
        # Get current breathing phase
        breathing_rhythm = self.biological_rhythms[RhythmType.BREATHING]
        phase = breathing_rhythm['phase']
        
        # Determine breathing state based on phase
        cycle_position = phase / (2 * math.pi)
        
        if 0.0 <= cycle_position < 0.4:
            # Inhale phase (0-40% of cycle)
            self.breathing_state = BreathingState.INHALE
        elif 0.4 <= cycle_position < 0.45:
            # Hold full (40-45% of cycle)
            self.breathing_state = BreathingState.HOLD_FULL
        elif 0.45 <= cycle_position < 0.85:
            # Exhale phase (45-85% of cycle)
            self.breathing_state = BreathingState.EXHALE
        else:
            # Hold empty (85-100% of cycle)
            self.breathing_state = BreathingState.HOLD_EMPTY
        
        # Check for accelerated breathing conditions
        if self.current_acceleration > 1.5:
            self.breathing_state = BreathingState.ACCELERATED
        
        # Check for deep breathing conditions
        if self.target_entropy > 0.8:
            self.breathing_state = BreathingState.DEEP

    async def _manage_adaptive_entropy(self, input_data: Dict[str, Any] = None):
        """Manage adaptive target entropy levels"""
        
        # Calculate desired entropy based on system state
        desired_entropy = self._calculate_desired_entropy(input_data)
        
        # Adapt target entropy gradually
        entropy_diff = desired_entropy - self.target_entropy
        self.target_entropy += entropy_diff * self.entropy_adaptation_rate
        
        # Clamp target entropy
        self.target_entropy = min(1.0, max(0.1, self.target_entropy))
        
        # Move current entropy toward target
        entropy_adjustment = (self.target_entropy - self.current_entropy) * 0.1
        self.current_entropy += entropy_adjustment
        
        # Add natural entropy variation based on breathing
        breathing_influence = self._get_breathing_entropy_influence()
        self.current_entropy += breathing_influence * 0.05
        
        # Clamp current entropy
        self.current_entropy = min(1.0, max(0.0, self.current_entropy))
        
        # Record entropy history
        self.entropy_history.append(self.current_entropy)

    def _calculate_desired_entropy(self, input_data: Dict[str, Any] = None) -> float:
        """Calculate desired entropy level based on system needs"""
        
        # Get system analytics
        analytics = self.consciousness_system.get_system_analytics()
        
        # Base entropy from system coherence
        coherence = analytics.get('current_coherence', 0.0)
        base_entropy = 0.6 - coherence * 0.2  # Higher coherence needs less entropy
        
        # Adjust based on adaptation efficiency
        adaptation_efficiency = analytics.get('adaptation_efficiency', 0.0)
        if adaptation_efficiency < 0.5:
            base_entropy += 0.2  # Need more entropy for better adaptation
        
        # Adjust based on input complexity
        if input_data:
            input_complexity = self._assess_input_complexity(input_data)
            base_entropy += input_complexity * 0.3
        
        # Adjust based on prediction accuracy
        prediction_accuracy = analytics.get('prediction_accuracy', 0.0)
        if prediction_accuracy < 0.6:
            base_entropy += 0.15  # Need more exploration
        
        return min(1.0, max(0.2, base_entropy))

    def _assess_input_complexity(self, input_data: Dict[str, Any]) -> float:
        """Assess complexity of input data"""
        
        complexity_factors = []
        
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                # Higher absolute values suggest more complexity
                complexity_factors.append(min(1.0, abs(float(value))))
            elif isinstance(value, str):
                # String length and content variety
                complexity_factors.append(min(1.0, len(value) / 50.0))
            elif isinstance(value, (list, tuple)):
                # List length and variance
                if value and all(isinstance(x, (int, float)) for x in value):
                    complexity_factors.append(min(1.0, np.var(value) + len(value) / 20.0))
        
        return np.mean(complexity_factors) if complexity_factors else 0.5

    def _get_breathing_entropy_influence(self) -> float:
        """Get entropy influence from current breathing state"""
        
        breathing_influences = {
            BreathingState.INHALE: 0.1,      # Slight entropy increase
            BreathingState.HOLD_FULL: -0.1,  # Entropy decrease (organization)
            BreathingState.EXHALE: 0.05,     # Mild entropy increase
            BreathingState.HOLD_EMPTY: -0.05, # Mild entropy decrease
            BreathingState.ACCELERATED: 0.3,  # High entropy increase
            BreathingState.DEEP: -0.2         # Entropy decrease (deep focus)
        }
        
        return breathing_influences.get(self.breathing_state, 0.0)

    def _calculate_biological_synchronization(self) -> float:
        """Calculate synchronization with biological rhythms"""
        
        # Calculate coherence between different rhythms
        coherence_factors = []
        
        # Primary rhythm coherence (breathing)
        breathing_rhythm = self.biological_rhythms[RhythmType.BREATHING]
        heartbeat_rhythm = self.biological_rhythms[RhythmType.HEARTBEAT]
        
        # Respiratory-cardiac coherence (Heart Rate Variability-like)
        phase_diff = abs(breathing_rhythm['phase'] - heartbeat_rhythm['phase'])
        phase_coherence = 1.0 - (phase_diff % (2 * math.pi)) / (2 * math.pi)
        coherence_factors.append(phase_coherence)
        
        # Neural rhythm coherence
        alpha_rhythm = self.biological_rhythms[RhythmType.NEURAL_ALPHA]
        theta_rhythm = self.biological_rhythms[RhythmType.NEURAL_THETA]
        
        # Alpha-theta coherence
        neural_phase_diff = abs(alpha_rhythm['phase'] - theta_rhythm['phase'])
        neural_coherence = 1.0 - (neural_phase_diff % (2 * math.pi)) / (2 * math.pi)
        coherence_factors.append(neural_coherence)
        
        # Circadian influence
        circadian_rhythm = self.biological_rhythms[RhythmType.CIRCADIAN]
        circadian_influence = 0.5 + 0.3 * math.cos(circadian_rhythm['phase'])
        coherence_factors.append(circadian_influence)
        
        return np.mean(coherence_factors)

    async def _apply_rhythmic_modulation(self):
        """Apply rhythmic modulation to consciousness system components"""
        
        # Modulate attention field based on breathing state
        if hasattr(self.consciousness_system, 'attention_field'):
            breathing_modulation = self._get_breathing_modulation()
            
            # Apply breathing influence to attention weights
            attention_weights = self.consciousness_system.attention_field.attention_weights
            modulated_weights = attention_weights * (1.0 + breathing_modulation * 0.1)
            self.consciousness_system.attention_field.attention_weights = modulated_weights
        
        # Modulate feedback loop adaptation rate based on entropy
        if hasattr(self.consciousness_system, 'feedback_loop'):
            entropy_modulation = self.current_entropy / self.target_entropy
            base_rate = self.consciousness_system.feedback_loop.base_adaptation_rate
            
            # Adapt learning rate based on entropy and breathing
            new_rate = base_rate * entropy_modulation * (1.0 + self.current_acceleration * 0.2)
            self.consciousness_system.feedback_loop.adaptation_rate = new_rate
        
        # Modulate fractal AI learning based on neural rhythms
        if hasattr(self.consciousness_system, 'fractal_ai'):
            alpha_phase = self.biological_rhythms[RhythmType.NEURAL_ALPHA]['phase']
            neural_modulation = 0.8 + 0.4 * math.sin(alpha_phase)
            
            # Apply neural modulation to learning rate
            for param_group in self.consciousness_system.fractal_ai.optimizer.param_groups:
                base_lr = param_group.get('base_lr', param_group['lr'])
                param_group['lr'] = base_lr * neural_modulation

    def _get_breathing_modulation(self) -> float:
        """Get modulation factor based on current breathing state"""
        
        breathing_modulations = {
            BreathingState.INHALE: 0.2,       # Increased activity/attention
            BreathingState.HOLD_FULL: 0.5,    # Peak activity
            BreathingState.EXHALE: -0.1,      # Decreased activity
            BreathingState.HOLD_EMPTY: -0.3,  # Minimum activity
            BreathingState.ACCELERATED: 0.8,  # High activity
            BreathingState.DEEP: 0.3           # Focused activity
        }
        
        return breathing_modulations.get(self.breathing_state, 0.0)

    async def _calculate_adaptive_acceleration(self, input_data: Dict[str, Any] = None) -> float:
        """Calculate adaptive acceleration factor for breathing"""
        
        # Check for acceleration triggers
        acceleration_triggers = []
        
        # High input complexity trigger
        if input_data:
            complexity = self._assess_input_complexity(input_data)
            if complexity > 0.7:
                acceleration_triggers.append(0.5)
        
        # Low adaptation efficiency trigger
        analytics = self.consciousness_system.get_system_analytics()
        adaptation_efficiency = analytics.get('adaptation_efficiency', 0.0)
        if adaptation_efficiency < 0.4:
            acceleration_triggers.append(0.6)
        
        # High error rate trigger
        prediction_accuracy = analytics.get('prediction_accuracy', 1.0)
        if prediction_accuracy < 0.5:
            acceleration_triggers.append(0.4)
        
        # Target entropy not being met
        entropy_gap = abs(self.target_entropy - self.current_entropy)
        if entropy_gap > 0.3:
            acceleration_triggers.append(entropy_gap)
        
        # Calculate acceleration factor
        if acceleration_triggers:
            target_acceleration = 1.0 + max(acceleration_triggers)
        else:
            target_acceleration = 1.0
        
        # Apply acceleration with decay
        self.current_acceleration = (
            self.current_acceleration * self.acceleration_decay +
            target_acceleration * (1 - self.acceleration_decay)
        )
        
        # Clamp acceleration
        self.current_acceleration = min(self.max_acceleration_factor, 
                                      max(1.0, self.current_acceleration))
        
        # Apply acceleration to breathing rate
        self.current_breathing_rate = self.base_breathing_rate * self.current_acceleration
        
        return self.current_acceleration

    def _calculate_rhythm_coherence(self) -> float:
        """Calculate overall rhythm coherence"""
        
        if len(self.rhythm_history) < 5:
            return 0.5
        
        # Analyze rhythm stability over recent history
        recent_rates = [m.breathing_rate for m in list(self.rhythm_history)[-10:]]
        recent_entropies = [m.entropy_level for m in list(self.rhythm_history)[-10:]]
        
        # Rate stability
        rate_variance = np.var(recent_rates) if len(recent_rates) > 1 else 0.0
        rate_coherence = max(0.0, 1.0 - rate_variance * 10.0)
        
        # Entropy stability
        entropy_variance = np.var(recent_entropies) if len(recent_entropies) > 1 else 0.0
        entropy_coherence = max(0.0, 1.0 - entropy_variance * 5.0)
        
        # Combined coherence
        overall_coherence = (rate_coherence * 0.6 + entropy_coherence * 0.4)
        
        return min(1.0, max(0.0, overall_coherence))

    def _calculate_breathing_amplitude(self) -> float:
        """Calculate current breathing amplitude for consciousness modulation"""
        
        breathing_rhythm = self.biological_rhythms[RhythmType.BREATHING]
        phase = breathing_rhythm['phase']
        amplitude = breathing_rhythm['amplitude']
        
        # Calculate sine wave amplitude with state modifications
        base_amplitude = amplitude * math.sin(phase)
        
        # Modify based on breathing state
        state_modifiers = {
            BreathingState.INHALE: 1.0,
            BreathingState.HOLD_FULL: 1.2,
            BreathingState.EXHALE: 0.8,
            BreathingState.HOLD_EMPTY: 0.6,
            BreathingState.ACCELERATED: 1.5,
            BreathingState.DEEP: 1.3
        }
        
        modifier = state_modifiers.get(self.breathing_state, 1.0)
        modulated_amplitude = base_amplitude * modifier
        
        # Apply acceleration influence
        accelerated_amplitude = modulated_amplitude * (1.0 + (self.current_acceleration - 1.0) * 0.3)
        
        return accelerated_amplitude

    def get_rhythmic_report(self) -> str:
        """Generate comprehensive rhythmic controller report"""
        
        if not self.rhythm_history:
            return "No rhythmic data available"
        
        latest_metrics = self.rhythm_history[-1]
        
        report = []
        report.append("🫁 RHYTHMIC CONTROLLER ENHANCEMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Breathing Status
        report.append(f"🌬️ BREATHING SYSTEM:")
        report.append(f"   • Current State: {latest_metrics.breathing_state.value.upper()}")
        report.append(f"   • Breathing Rate: {latest_metrics.breathing_rate:.3f} Hz")
        report.append(f"   • Base Rate: {self.base_breathing_rate:.3f} Hz")
        report.append(f"   • Acceleration Factor: {latest_metrics.adaptive_acceleration:.2f}x")
        report.append(f"   • Amplitude: {latest_metrics.consciousness_breathing_amplitude:.3f}")
        report.append("")
        
        # Entropy Management
        report.append(f"🎯 ENTROPY MANAGEMENT:")
        report.append(f"   • Current Entropy: {latest_metrics.entropy_level:.3f}")
        report.append(f"   • Target Entropy: {latest_metrics.target_entropy:.3f}")
        entropy_status = "✅ ON TARGET" if abs(latest_metrics.entropy_level - latest_metrics.target_entropy) < 0.1 else "🎯 ADJUSTING"
        report.append(f"   • Status: {entropy_status}")
        report.append("")
        
        # Biological Synchronization
        report.append(f"🧬 BIOLOGICAL SYNCHRONIZATION:")
        report.append(f"   • Rhythm Coherence: {latest_metrics.rhythm_coherence:.3f}")
        report.append(f"   • Bio-Sync Level: {latest_metrics.biological_synchronization:.3f}")
        
        # Rhythm Details
        report.append("")
        report.append("📊 BIOLOGICAL RHYTHMS:")
        for rhythm_type, params in self.biological_rhythms.items():
            phase_degrees = (params['phase'] / (2 * math.pi)) * 360
            report.append(f"   • {rhythm_type.value.title()}: {params['frequency']:.3f} Hz (Phase: {phase_degrees:.1f}°)")
        
        # Performance Indicators
        report.append("")
        report.append("⚡ PERFORMANCE INDICATORS:")
        
        if latest_metrics.rhythm_coherence > 0.7:
            report.append("   ✅ High rhythm coherence - stable biological integration")
        elif latest_metrics.rhythm_coherence > 0.4:
            report.append("   ⚠️ Moderate rhythm coherence - some variability")
        else:
            report.append("   ❌ Low rhythm coherence - needs stabilization")
        
        if latest_metrics.adaptive_acceleration > 2.0:
            report.append("   🚀 High acceleration mode - responding to complex demands")
        elif latest_metrics.adaptive_acceleration > 1.2:
            report.append("   ⚡ Moderate acceleration - adapting to challenges")
        else:
            report.append("   🌊 Normal rhythm - stable operation")
        
        return "\n".join(report)

# Integration function
async def integrate_rhythmic_controller(consciousness_system):
    """Integrate Rhythmic Controller Enhancement with consciousness system"""
    
    logger.info("🫁 Integrating Rhythmic Controller Enhancement")
    
    # Create rhythmic controller
    rhythmic_controller = RhythmicController(consciousness_system)
    
    # Run initial rhythmic cycles
    for cycle in range(10):
        test_input = {
            'complexity_level': 0.3 + cycle * 0.05,
            'attention_demand': 0.4 + cycle * 0.06,
            'adaptation_pressure': 0.2 + cycle * 0.08,
            'cycle': cycle
        }
        
        metrics = await rhythmic_controller.process_rhythmic_cycle(test_input, time_delta=0.1)
        logger.info(f"Cycle {cycle + 1}: {metrics.breathing_state.value} | "
                   f"Rate: {metrics.breathing_rate:.3f} Hz | "
                   f"Entropy: {metrics.entropy_level:.3f}")
    
    # Generate and display report
    report = rhythmic_controller.get_rhythmic_report()
    print("\n" + report)
    
    return rhythmic_controller

if __name__ == "__main__":
    print("🫁 Rhythmic Controller Enhancement System Ready")
    print("Use: rhythmic_controller = await integrate_rhythmic_controller(consciousness_system)")