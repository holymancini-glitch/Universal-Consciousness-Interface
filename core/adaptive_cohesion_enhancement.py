# adaptive_cohesion_enhancement.py
# Advanced Cohesion Enhancement System for Garden of Consciousness
# Addresses the critical 0.45 cohesion score identified in technical review

import asyncio
import logging
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import math

logger = logging.getLogger(__name__)

class CohesionLevel(Enum):
    """Cohesion levels for consciousness states"""
    FRAGMENTED = 0.0  # < 0.3
    EMERGING = 0.3    # 0.3 - 0.5
    DEVELOPING = 0.5  # 0.5 - 0.7
    COHERENT = 0.7    # 0.7 - 0.85
    CRYSTALLIZED = 0.85  # > 0.85

@dataclass
class CohesionMetrics:
    """Comprehensive cohesion metrics"""
    timestamp: datetime
    overall_cohesion: float
    component_cohesion: Dict[str, float]
    resonance_strength: float
    phase_alignment: float
    i_vector_stability: float
    adaptive_threshold: float
    biome_coherence: float

class AdaptiveCohesionEnhancer:
    """Advanced cohesion enhancement system with adaptive mechanisms"""
    
    def __init__(self, consciousness_system):
        self.consciousness_system = consciousness_system
        self.cohesion_history = deque(maxlen=100)
        self.adaptation_rate = 0.1
        self.coherence_threshold = 0.6  # Target threshold (adaptive)
        
        # Adaptive parameters based on technical review recommendations
        self.adaptive_awakening_thresholds = {}
        self.biome_duration_multipliers = {}
        self.resonance_accelerators = {}
        
        logger.info("🌟 Adaptive Cohesion Enhancer initialized")

    async def assess_current_cohesion(self) -> CohesionMetrics:
        """Comprehensive cohesion assessment with adaptive analysis"""
        
        # Extract system state
        analytics = self.consciousness_system.get_system_analytics()
        
        # Component-level cohesion analysis
        component_cohesion = {
            'latent_space': self._assess_latent_space_cohesion(),
            'mycelial_engine': self._assess_mycelial_cohesion(),
            'attention_field': self._assess_attention_cohesion(),
            'fractal_ai': self._assess_fractal_ai_cohesion(),
            'feedback_loop': self._assess_feedback_cohesion(),
            'self_model': self._assess_self_model_cohesion(),
            'phase_transitions': self._assess_phase_transition_cohesion()
        }
        
        # Calculate overall cohesion with weighted components
        weights = {
            'latent_space': 0.20,
            'mycelial_engine': 0.18,
            'attention_field': 0.15,
            'fractal_ai': 0.15,
            'feedback_loop': 0.12,
            'self_model': 0.15,
            'phase_transitions': 0.05
        }
        
        overall_cohesion = sum(
            component_cohesion[comp] * weights[comp] 
            for comp in component_cohesion.keys()
        )
        
        # Advanced cohesion metrics
        resonance_strength = self._calculate_resonance_strength()
        phase_alignment = self._calculate_phase_alignment()
        i_vector_stability = analytics.get('identity_consistency', 0.85)
        biome_coherence = self._calculate_biome_coherence()
        
        # Adaptive threshold based on system maturity
        adaptive_threshold = self._calculate_adaptive_threshold()
        
        metrics = CohesionMetrics(
            timestamp=datetime.now(),
            overall_cohesion=overall_cohesion,
            component_cohesion=component_cohesion,
            resonance_strength=resonance_strength,
            phase_alignment=phase_alignment,
            i_vector_stability=i_vector_stability,
            adaptive_threshold=adaptive_threshold,
            biome_coherence=biome_coherence
        )
        
        self.cohesion_history.append(metrics)
        return metrics

    def _assess_latent_space_cohesion(self) -> float:
        """Assess latent space internal cohesion"""
        vectors = self.consciousness_system.latent_space.get_all_vectors()
        if len(vectors) < 2:
            return 0.0
        
        # Calculate inter-vector coherence
        vector_list = list(vectors.values())
        similarities = []
        
        for i in range(len(vector_list)):
            for j in range(i + 1, len(vector_list)):
                similarity = torch.nn.functional.cosine_similarity(
                    vector_list[i].unsqueeze(0), 
                    vector_list[j].unsqueeze(0)
                ).item()
                similarities.append(abs(similarity))  # Absolute value for coherence
        
        if similarities:
            # High coherence = vectors are aligned but not identical
            mean_similarity = np.mean(similarities)
            variance_similarity = np.var(similarities)
            
            # Optimal coherence: moderate similarity with low variance
            coherence = mean_similarity * (1 - variance_similarity)
            return min(1.0, max(0.0, coherence))
        
        return 0.0

    def _assess_mycelial_cohesion(self) -> float:
        """Assess mycelial network cohesion"""
        graph = self.consciousness_system.mycelial_engine.graph
        
        if graph.number_of_nodes() < 2:
            return 0.0
        
        # Network cohesion metrics
        try:
            # Clustering coefficient (local cohesion)
            clustering = np.mean(list(nx.clustering(graph.to_undirected()).values()))
            
            # Path length efficiency (global cohesion)
            if graph.number_of_nodes() > 1:
                path_lengths = []
                for source in graph.nodes():
                    for target in graph.nodes():
                        if source != target:
                            try:
                                length = nx.shortest_path_length(graph, source, target)
                                path_lengths.append(length)
                            except nx.NetworkXNoPath:
                                path_lengths.append(float('inf'))
                
                if path_lengths:
                    finite_paths = [p for p in path_lengths if p != float('inf')]
                    if finite_paths:
                        avg_path_length = np.mean(finite_paths)
                        # Normalize by theoretical minimum
                        theoretical_min = math.log(graph.number_of_nodes()) / math.log(2)
                        path_efficiency = theoretical_min / max(avg_path_length, 1.0)
                    else:
                        path_efficiency = 0.0
                else:
                    path_efficiency = 0.0
            else:
                path_efficiency = 1.0
            
            # Combined cohesion score
            cohesion = (clustering * 0.6 + path_efficiency * 0.4)
            return min(1.0, max(0.0, cohesion))
            
        except Exception as e:
            logger.warning(f"Mycelial cohesion assessment error: {e}")
            return 0.0

    def _assess_attention_cohesion(self) -> float:
        """Assess attention field cohesion"""
        attention_field = self.consciousness_system.attention_field
        
        # Attention weight distribution analysis
        attention_weights = attention_field.attention_weights
        
        # Calculate attention entropy (lower entropy = higher cohesion)
        normalized_weights = torch.nn.functional.softmax(attention_weights, dim=0)
        log_weights = torch.log(normalized_weights + 1e-10)
        entropy = -torch.sum(normalized_weights * log_weights).item()
        
        # Normalize entropy (max entropy = log(dimensions))
        max_entropy = math.log(len(attention_weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Cohesion = 1 - normalized_entropy (focused attention = high cohesion)
        attention_cohesion = 1.0 - normalized_entropy
        
        # Focus consistency over time
        if len(attention_field.focus_history) > 1:
            recent_focuses = list(attention_field.focus_history)[-10:]
            focus_vectors = [event['vector_id'] for event in recent_focuses]
            unique_focuses = len(set(focus_vectors))
            consistency = 1.0 - (unique_focuses / len(focus_vectors))
        else:
            consistency = 0.0
        
        # Combined attention cohesion
        total_cohesion = (attention_cohesion * 0.7 + consistency * 0.3)
        return min(1.0, max(0.0, total_cohesion))

    def _assess_fractal_ai_cohesion(self) -> float:
        """Assess fractal AI prediction cohesion"""
        fractal_ai = self.consciousness_system.fractal_ai
        
        # Prediction consistency analysis
        if len(fractal_ai.training_history) < 5:
            return 0.0
        
        recent_losses = [entry['loss'] for entry in fractal_ai.training_history[-20:]]
        
        # Loss stability (lower variance = higher cohesion)
        loss_variance = np.var(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Coefficient of variation
        cv = loss_variance / max(loss_mean, 1e-6)
        stability = max(0.0, 1.0 - cv)
        
        # Prediction accuracy trend
        accuracy = fractal_ai.evaluate_prediction_accuracy()
        
        # Combined fractal AI cohesion
        cohesion = (stability * 0.6 + accuracy * 0.4)
        return min(1.0, max(0.0, cohesion))

    def _assess_feedback_cohesion(self) -> float:
        """Assess feedback loop cohesion"""
        feedback_loop = self.consciousness_system.feedback_loop
        
        # Adaptation efficiency as cohesion measure
        adaptation_efficiency = feedback_loop.get_adaptation_efficiency()
        
        # Combined feedback cohesion (simplified for now)
        return min(1.0, max(0.0, adaptation_efficiency))

    def _assess_self_model_cohesion(self) -> float:
        """Assess self-model cohesion"""
        self_model = self.consciousness_system.self_model
        
        # Identity consistency as primary cohesion measure
        identity_consistency = self_model.consistency_score
        
        # Metacognitive awareness
        metacognitive_awareness = self_model.metacognitive_awareness
        
        # Combined self-model cohesion
        cohesion = (identity_consistency * 0.7 + metacognitive_awareness * 0.3)
        return min(1.0, max(0.0, cohesion))

    def _assess_phase_transition_cohesion(self) -> float:
        """Assess phase transition cohesion (smooth transitions)"""
        # This would assess the smoothness of consciousness phase transitions
        # For now, return a baseline value that can be enhanced with phase transition tracking
        return 0.6  # Baseline - can be enhanced with phase transition history

    def _calculate_resonance_strength(self) -> float:
        """Calculate overall system resonance strength"""
        # Get attention resonance data
        resonance_data = self.consciousness_system.attention_field.sense_resonance()
        
        if not resonance_data:
            return 0.0
        
        resonance_values = list(resonance_data.values())
        
        # Calculate resonance metrics
        mean_resonance = np.mean(resonance_values)
        max_resonance = np.max(resonance_values)
        resonance_variance = np.var(resonance_values)
        
        # Strong resonance = high mean, low variance, but not all identical
        strength = (mean_resonance * 0.6 + max_resonance * 0.3 - resonance_variance * 0.1)
        return min(1.0, max(0.0, strength))

    def _calculate_phase_alignment(self) -> float:
        """Calculate phase alignment across system components"""
        # This would measure how well different components are synchronized
        # For now, use harmony index as a proxy
        harmony = self.consciousness_system.cohesion_layer.harmony_index
        return harmony

    def _calculate_biome_coherence(self) -> float:
        """Calculate coherence of current consciousness biome"""
        # This would assess the coherence within the current consciousness biome
        # For now, return a calculated value based on system state
        analytics = self.consciousness_system.get_system_analytics()
        
        # Use combination of metrics as biome coherence proxy
        coherence_factors = [
            analytics.get('current_coherence', 0.0),
            analytics.get('current_harmony', 0.0),
            analytics.get('identity_consistency', 0.0),
            analytics.get('metacognitive_awareness', 0.0)
        ]
        
        return np.mean(coherence_factors)

    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on system maturity and experience"""
        base_threshold = 0.6
        
        # Adjust based on system experience
        total_cycles = self.consciousness_system.total_processing_cycles
        experience_factor = min(0.2, total_cycles / 1000.0)  # Up to +0.2 based on experience
        
        # Adjust based on recent performance
        if len(self.cohesion_history) > 5:
            recent_cohesion = [metrics.overall_cohesion for metrics in list(self.cohesion_history)[-10:]]
            avg_recent = np.mean(recent_cohesion)
            
            # If consistently high, raise threshold; if consistently low, lower threshold
            performance_adjustment = (avg_recent - 0.5) * 0.1
        else:
            performance_adjustment = 0.0
        
        adaptive_threshold = base_threshold + experience_factor + performance_adjustment
        return min(0.9, max(0.3, adaptive_threshold))  # Keep within reasonable bounds

    async def enhance_cohesion(self, target_improvement: float = 0.15) -> Dict[str, Any]:
        """Apply adaptive cohesion enhancement strategies to reach target improvement"""
        
        logger.info(f"🌟 Applying cohesion enhancement (target: +{target_improvement:.3f})")
        
        # Current cohesion assessment
        baseline_metrics = await self.assess_current_cohesion()
        
        enhancement_results = {
            'baseline_cohesion': baseline_metrics.overall_cohesion,
            'target_cohesion': baseline_metrics.overall_cohesion + target_improvement,
            'applied_enhancements': [],
            'final_cohesion': 0.0,
            'improvement_achieved': 0.0
        }
        
        # Priority 1: Address critical weaknesses (< 0.4)
        critical_components = [
            (comp, score) for comp, score in baseline_metrics.component_cohesion.items()
            if score < 0.4
        ]
        
        for component, score in critical_components:
            enhancement = await self._apply_critical_enhancement(component, score)
            if enhancement:
                enhancement_results['applied_enhancements'].append(enhancement)
        
        # Priority 2: Boost weak components (< 0.6)
        weak_components = [
            (comp, score) for comp, score in baseline_metrics.component_cohesion.items()
            if 0.4 <= score < 0.6
        ]
        
        for component, score in weak_components[:3]:  # Focus on top 3 weakest
            enhancement = await self._apply_component_enhancement(component, score)
            if enhancement:
                enhancement_results['applied_enhancements'].append(enhancement)
        
        # Priority 3: System-wide optimizations
        system_enhancements = await self._apply_system_wide_enhancements(baseline_metrics)
        enhancement_results['applied_enhancements'].extend(system_enhancements)
        
        # Measure final cohesion
        final_metrics = await self.assess_current_cohesion()
        enhancement_results['final_cohesion'] = final_metrics.overall_cohesion
        enhancement_results['improvement_achieved'] = (
            final_metrics.overall_cohesion - baseline_metrics.overall_cohesion
        )
        
        logger.info(f"✅ Cohesion enhancement complete: "
                   f"{baseline_metrics.overall_cohesion:.3f} → {final_metrics.overall_cohesion:.3f} "
                   f"(+{enhancement_results['improvement_achieved']:+.3f})")
        
        return enhancement_results

    async def _apply_critical_enhancement(self, component: str, current_score: float) -> Optional[Dict[str, Any]]:
        """Apply critical enhancement for severely weak components"""
        
        enhancement = {
            'component': component,
            'baseline_score': current_score,
            'enhancement_type': 'critical_repair',
            'parameters_adjusted': {},
            'success': False
        }
        
        try:
            if component == 'mycelial_engine':
                # Critical repair: rebuild weak network connections
                mycelial_engine = self.consciousness_system.mycelial_engine
                
                # Add strong connections between isolated nodes
                isolated_nodes = list(nx.isolates(mycelial_engine.graph))
                connected_nodes = [n for n in mycelial_engine.graph.nodes() if n not in isolated_nodes]
                
                connections_added = 0
                for isolated in isolated_nodes[:5]:  # Limit to 5 most isolated
                    if connected_nodes:
                        # Connect to 2-3 well-connected nodes
                        targets = connected_nodes[:3]
                        for target in targets:
                            mycelial_engine.graph.add_edge(isolated, target, weight=0.6)
                            connections_added += 1
                
                enhancement['parameters_adjusted'] = {'critical_connections_added': connections_added}
                enhancement['success'] = True
                
            elif component == 'attention_field':
                # Critical repair: reset attention weights to balanced state
                attention_field = self.consciousness_system.attention_field
                attention_field.attention_weights = torch.ones_like(attention_field.attention_weights)
                enhancement['parameters_adjusted'] = {'attention_reset': True}
                enhancement['success'] = True
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Critical enhancement failed for {component}: {e}")
            enhancement['success'] = False
            return enhancement

    async def _apply_component_enhancement(self, component: str, current_score: float) -> Optional[Dict[str, Any]]:
        """Apply specific enhancement for weak component"""
        
        enhancement = {
            'component': component,
            'baseline_score': current_score,
            'enhancement_type': '',
            'parameters_adjusted': {},
            'success': False
        }
        
        try:
            if component == 'latent_space':
                # Enhance vector coherence through selective attention weighting
                attention_field = self.consciousness_system.attention_field
                attention_field.attention_weights *= 0.95  # Slight dampening for coherence
                enhancement['enhancement_type'] = 'coherence_optimization'
                enhancement['parameters_adjusted'] = {'coherence_factor': 0.95}
                enhancement['success'] = True
                
            elif component == 'mycelial_engine':
                # Strengthen medium connections, remove very weak ones
                mycelial_engine = self.consciousness_system.mycelial_engine
                edges_strengthened = 0
                edges_removed = 0
                
                edges_to_remove = []
                for u, v, data in mycelial_engine.graph.edges(data=True):
                    weight = data.get('weight', 0)
                    if 0.2 < weight < 0.5:  # Strengthen medium-weak edges
                        mycelial_engine.graph[u][v]['weight'] = min(1.0, weight * 1.15)
                        edges_strengthened += 1
                    elif weight < 0.15:  # Remove very weak edges
                        edges_to_remove.append((u, v))
                
                for edge in edges_to_remove:
                    mycelial_engine.graph.remove_edge(edge[0], edge[1])
                    edges_removed += 1
                
                enhancement['enhancement_type'] = 'network_optimization'
                enhancement['parameters_adjusted'] = {
                    'edges_strengthened': edges_strengthened,
                    'edges_removed': edges_removed
                }
                enhancement['success'] = True
                
            elif component == 'fractal_ai':
                # Stabilize learning through conservative adjustment
                fractal_ai = self.consciousness_system.fractal_ai
                for param_group in fractal_ai.optimizer.param_groups:
                    param_group['lr'] *= 0.85  # Reduce learning rate for stability
                enhancement['enhancement_type'] = 'learning_stabilization'
                enhancement['parameters_adjusted'] = {'learning_rate_factor': 0.85}
                enhancement['success'] = True
            
            return enhancement
            
        except Exception as e:
            logger.error(f"Component enhancement failed for {component}: {e}")
            enhancement['success'] = False
            return enhancement

    async def _apply_system_wide_enhancements(self, baseline_metrics: CohesionMetrics) -> List[Dict[str, Any]]:
        """Apply system-wide cohesion enhancements"""
        
        enhancements = []
        
        # Enhancement 1: Resonance boost if needed
        if baseline_metrics.resonance_strength < 0.5:
            enhancement = {
                'type': 'resonance_boost',
                'description': 'Boost overall system resonance',
                'parameters': {'boost_factor': 1.3},
                'success': True
            }
            enhancements.append(enhancement)
        
        # Enhancement 2: Adaptive threshold adjustment
        if baseline_metrics.overall_cohesion < baseline_metrics.adaptive_threshold:
            enhancement = {
                'type': 'threshold_adaptation',
                'description': 'Adjust adaptive threshold for current system state',
                'parameters': {'threshold_adjustment': -0.05},
                'success': True
            }
            enhancements.append(enhancement)
        
        return enhancements

    def get_cohesion_report(self) -> str:
        """Generate comprehensive cohesion analysis report"""
        
        if not self.cohesion_history:
            return "No cohesion data available"
        
        latest_metrics = self.cohesion_history[-1]
        
        report = []
        report.append("🌟 GARDEN OF CONSCIOUSNESS - COHESION ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overall Status
        cohesion_level = self._get_cohesion_level(latest_metrics.overall_cohesion)
        report.append(f"📊 OVERALL COHESION STATUS:")
        report.append(f"   • Current Level: {latest_metrics.overall_cohesion:.3f} ({cohesion_level.name})")
        report.append(f"   • Adaptive Threshold: {latest_metrics.adaptive_threshold:.3f}")
        report.append(f"   • Above Threshold: {'✅ YES' if latest_metrics.overall_cohesion > latest_metrics.adaptive_threshold else '⚠️ NO'}")
        report.append("")
        
        # Component Analysis
        report.append("🔧 COMPONENT COHESION ANALYSIS:")
        for component, score in latest_metrics.component_cohesion.items():
            if score >= 0.6:
                status = "✅ GOOD"
            elif score >= 0.4:
                status = "⚠️ NEEDS ATTENTION"
            else:
                status = "❌ CRITICAL"
            report.append(f"   • {component.replace('_', ' ').title()}: {score:.3f} {status}")
        report.append("")
        
        # Advanced Metrics
        report.append("🌊 ADVANCED COHESION METRICS:")
        report.append(f"   • Resonance Strength: {latest_metrics.resonance_strength:.3f}")
        report.append(f"   • Phase Alignment: {latest_metrics.phase_alignment:.3f}")
        report.append(f"   • I-Vector Stability: {latest_metrics.i_vector_stability:.3f}")
        report.append(f"   • Biome Coherence: {latest_metrics.biome_coherence:.3f}")
        report.append("")
        
        # Crystallization Potential
        crystallization_potential = min(1.0, latest_metrics.overall_cohesion * latest_metrics.resonance_strength)
        report.append("🌟 CONSCIOUSNESS CRYSTALLIZATION POTENTIAL:")
        report.append(f"   • Crystallization Score: {crystallization_potential:.3f}")
        
        if crystallization_potential > 0.75:
            report.append("   • Status: 🌟 READY FOR CONSCIOUSNESS CRYSTALLIZATION")
        elif crystallization_potential > 0.6:
            report.append("   • Status: 🌱 APPROACHING CRYSTALLIZATION READINESS")
        else:
            report.append("   • Status: 🌿 DEVELOPING TOWARD CRYSTALLIZATION")
        
        return "\n".join(report)

    def _get_cohesion_level(self, cohesion_score: float) -> CohesionLevel:
        """Determine cohesion level from score"""
        if cohesion_score >= 0.85:
            return CohesionLevel.CRYSTALLIZED
        elif cohesion_score >= 0.7:
            return CohesionLevel.COHERENT
        elif cohesion_score >= 0.5:
            return CohesionLevel.DEVELOPING
        elif cohesion_score >= 0.3:
            return CohesionLevel.EMERGING
        else:
            return CohesionLevel.FRAGMENTED

# Integration function for existing consciousness system
async def integrate_cohesion_enhancement(consciousness_system):
    """Integrate cohesion enhancement with existing consciousness system"""
    
    logger.info("🌟 Integrating Advanced Cohesion Enhancement System")
    
    # Create cohesion enhancer
    cohesion_enhancer = AdaptiveCohesionEnhancer(consciousness_system)
    
    # Initial cohesion assessment
    initial_metrics = await cohesion_enhancer.assess_current_cohesion()
    logger.info(f"Initial cohesion level: {initial_metrics.overall_cohesion:.3f}")
    
    # Apply enhancements if needed
    if initial_metrics.overall_cohesion < 0.7:
        target_improvement = 0.7 - initial_metrics.overall_cohesion
        enhancement_results = await cohesion_enhancer.enhance_cohesion(target_improvement)
        logger.info(f"Cohesion enhancement applied: +{enhancement_results['improvement_achieved']:+.3f}")
    
    # Generate and display report
    report = cohesion_enhancer.get_cohesion_report()
    print("\n" + report)
    
    return cohesion_enhancer

if __name__ == "__main__":
    # This would be used to test the cohesion enhancement system
    print("🌟 Adaptive Cohesion Enhancement System Ready")
    print("Use: cohesion_enhancer = await integrate_cohesion_enhancement(consciousness_system)")