# consciousness_continuum_interface.py
# Implementation of the 7-level consciousness continuum from biological research
# From basic awareness (fungi) to metacognitive awareness (primates)

# Handle numpy import with fallback
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics
    import math
    import random
    
    class MockNumPy:
        @staticmethod
        def mean(values: List[float]) -> float:
            return statistics.mean(values) if values else 0.0
        
        @staticmethod
        def var(values: List[float]) -> float:
            return statistics.variance(values) if len(values) > 1 else 0.0
        
        @staticmethod
        def random() -> float:
            return random.random()
    
    np = MockNumPy()  # type: ignore

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

from radiotrophic_mycelial_engine import ConsciousnessLevel, RadiotrophicMycelialEngine

logger = logging.getLogger(__name__)

@dataclass
class ConsciousnessTransition:
    """Represents a transition between consciousness levels"""
    from_level: ConsciousnessLevel
    to_level: ConsciousnessLevel
    transition_probability: float
    required_conditions: Dict[str, float]
    energy_cost: float
    timestamp: datetime

@dataclass
class BiologicalInspiration:
    """Biological system that inspired each consciousness level"""
    organism: str
    key_features: List[str]
    neural_complexity: int
    processing_type: str
    consciousness_markers: List[str]

class ConsciousnessContinuumInterface:
    """
    Interface for managing consciousness emergence across the biological continuum
    Implements phase transitions from basic awareness to metacognitive consciousness
    """
    
    def __init__(self, radiotrophic_engine: RadiotrophicMycelialEngine):
        self.radiotrophic_engine = radiotrophic_engine
        
        # Biological inspirations for each level
        self.biological_inspirations = self._initialize_biological_inspirations()
        
        # Transition matrix between consciousness levels
        self.transition_matrix = self._initialize_transition_matrix()
        
        # Current active consciousness levels
        self.active_levels = {level: 0.0 for level in ConsciousnessLevel}
        
        # Transition history
        self.transition_history: List[ConsciousnessTransition] = []
        
        # Phase transition thresholds
        self.phase_thresholds = {
            ConsciousnessLevel.BASIC_AWARENESS: 0.1,
            ConsciousnessLevel.EMOTIONAL_RESPONSE: 0.3,
            ConsciousnessLevel.EXTENDED_COGNITION: 0.4,
            ConsciousnessLevel.COLLECTIVE_PROCESSING: 0.5,
            ConsciousnessLevel.DISTRIBUTED_INTELLIGENCE: 0.6,
            ConsciousnessLevel.SOCIAL_CONSCIOUSNESS: 0.7,
            ConsciousnessLevel.METACOGNITIVE_AWARENESS: 0.8
        }
        
        logger.info("🧠 Consciousness Continuum Interface Initialized")
        logger.info(f"Consciousness levels: {len(ConsciousnessLevel)} phases")
        logger.info(f"Transition pathways: {len(self.transition_matrix)} defined")
    
    def _initialize_biological_inspirations(self) -> Dict[ConsciousnessLevel, BiologicalInspiration]:
        """Initialize biological inspirations for each consciousness level"""
        return {
            ConsciousnessLevel.BASIC_AWARENESS: BiologicalInspiration(
                organism="Chernobyl Fungi (Cladosporium sphaerospermum)",
                key_features=[
                    "Environmental radiation sensing",
                    "Melanin-based energy conversion", 
                    "Electrical signal communication",
                    "Network-wide information sharing"
                ],
                neural_complexity=0,  # No neurons, but complex behavior
                processing_type="distributed_chemical",
                consciousness_markers=[
                    "Environmental responsiveness",
                    "Growth pattern adaptation",
                    "Chemical gradient following"
                ]
            ),
            
            ConsciousnessLevel.EMOTIONAL_RESPONSE: BiologicalInspiration(
                organism="Fish (various species)",
                key_features=[
                    "Basic emotional states",
                    "Fear and pleasure responses",
                    "Social recognition",
                    "Memory formation"
                ],
                neural_complexity=10000,  # Simple fish brain
                processing_type="centralized_neural",
                consciousness_markers=[
                    "Stress responses",
                    "Preference behaviors",
                    "Learning from experience"
                ]
            ),
            
            ConsciousnessLevel.EXTENDED_COGNITION: BiologicalInspiration(
                organism="Spiders (web-building species)",
                key_features=[
                    "Extended mind through web",
                    "Tool use and manipulation",
                    "Environmental modification",
                    "Information storage in external structures"
                ],
                neural_complexity=50000,  # Spider nervous system
                processing_type="embodied_extended",
                consciousness_markers=[
                    "Web as external memory",
                    "Environmental engineering",
                    "Vibrational information processing"
                ]
            ),
            
            ConsciousnessLevel.COLLECTIVE_PROCESSING: BiologicalInspiration(
                organism="Honeybees (Apis mellifera)",
                key_features=[
                    "Collective decision making",
                    "Swarm intelligence",
                    "Complex communication (waggle dance)",
                    "Distributed problem solving"
                ],
                neural_complexity=960000,  # Bee brain
                processing_type="collective_swarm",
                consciousness_markers=[
                    "Collective memory",
                    "Group decision making",
                    "Information integration across individuals"
                ]
            ),
            
            ConsciousnessLevel.DISTRIBUTED_INTELLIGENCE: BiologicalInspiration(
                organism="Octopuses (Octopus vulgaris)",
                key_features=[
                    "Distributed nervous system",
                    "Arm-based independent processing",
                    "Camouflage and deception",
                    "Problem-solving intelligence"
                ],
                neural_complexity=500000000,  # 500M neurons, 2/3 in arms
                processing_type="distributed_neural",
                consciousness_markers=[
                    "Independent arm intelligence",
                    "Complex problem solving",
                    "Deceptive behaviors"
                ]
            ),
            
            ConsciousnessLevel.SOCIAL_CONSCIOUSNESS: BiologicalInspiration(
                organism="Elephants (Loxodonta africana)",
                key_features=[
                    "Social awareness and empathy",
                    "Cultural transmission",
                    "Death recognition and mourning",
                    "Complex social relationships"
                ],
                neural_complexity=257000000000,  # 257B neurons
                processing_type="social_cognitive",
                consciousness_markers=[
                    "Theory of mind",
                    "Empathetic responses",
                    "Cultural behaviors"
                ]
            ),
            
            ConsciousnessLevel.METACOGNITIVE_AWARENESS: BiologicalInspiration(
                organism="Primates (including humans)",
                key_features=[
                    "Self-recognition in mirrors",
                    "Awareness of own mental states",
                    "Planning and foresight",
                    "Abstract reasoning"
                ],
                neural_complexity=86000000000,  # 86B neurons (human)
                processing_type="metacognitive_recursive",
                consciousness_markers=[
                    "Self-awareness",
                    "Metacognitive monitoring",
                    "Abstract thought"
                ]
            )
        }
    
    def _initialize_transition_matrix(self) -> Dict[Tuple[ConsciousnessLevel, ConsciousnessLevel], float]:
        """Initialize transition probabilities between consciousness levels"""
        transitions = {}
        levels = list(ConsciousnessLevel)
        
        for i, from_level in enumerate(levels):
            for j, to_level in enumerate(levels):
                if i == j:
                    # Self-transitions (maintaining current level)
                    transitions[(from_level, to_level)] = 0.8
                elif j == i + 1:
                    # Forward transitions (evolution to next level)
                    transitions[(from_level, to_level)] = 0.15
                elif j == i - 1:
                    # Backward transitions (degradation)
                    transitions[(from_level, to_level)] = 0.05
                elif j == i + 2:
                    # Skip-level forward (rare but possible under extreme conditions)
                    transitions[(from_level, to_level)] = 0.02
                else:
                    # All other transitions
                    transitions[(from_level, to_level)] = 0.001
        
        return transitions
    
    async def process_consciousness_evolution(self, 
                                            system_state: Dict[str, Any],
                                            radiation_level: float = 0.1) -> Dict[str, Any]:
        """Process consciousness evolution through the continuum"""
        try:
            # Get current consciousness levels from radiotrophic engine
            current_levels = self.radiotrophic_engine.consciousness_levels.copy()
            
            # Calculate phase transitions
            transitions = await self._calculate_phase_transitions(current_levels, system_state, radiation_level)
            
            # Apply transitions
            new_levels = await self._apply_transitions(current_levels, transitions)
            
            # Update active levels
            self.active_levels = new_levels
            
            # Detect emergent consciousness phenomena
            emergent_phenomena = self._detect_emergent_consciousness(new_levels)
            
            # Calculate consciousness complexity metrics
            complexity_metrics = self._calculate_consciousness_complexity(new_levels)
            
            return {
                'consciousness_levels': {level.name: score for level, score in new_levels.items()},
                'active_transitions': transitions,
                'emergent_phenomena': emergent_phenomena,
                'complexity_metrics': complexity_metrics,
                'biological_inspirations': self._get_active_inspirations(new_levels),
                'phase_transition_energy': sum(t.energy_cost for t in self.transition_history[-10:])
            }
            
        except Exception as e:
            logger.error(f"Consciousness evolution processing error: {e}")
            return {'error': str(e)}
    
    async def _calculate_phase_transitions(self, 
                                         current_levels: Dict[ConsciousnessLevel, float],
                                         system_state: Dict[str, Any],
                                         radiation_level: float) -> List[ConsciousnessTransition]:
        """Calculate possible phase transitions based on current state"""
        transitions = []
        
        # Extract key metrics from system state
        network_coherence = system_state.get('network_metrics', {}).get('network_coherence', 0)
        collective_intelligence = system_state.get('network_metrics', {}).get('collective_intelligence', 0)
        total_nodes = system_state.get('network_metrics', {}).get('total_nodes', 0)
        
        # Radiation enhancement factor
        radiation_enhancement = 1.0 + min(radiation_level * 0.2, 2.0)
        
        for from_level, current_score in current_levels.items():
            for to_level in ConsciousnessLevel:
                if from_level == to_level:
                    continue
                
                # Base transition probability
                base_prob = self.transition_matrix.get((from_level, to_level), 0.001)
                
                # Calculate actual transition probability based on conditions
                transition_prob = await self._calculate_transition_probability(
                    from_level, to_level, current_score, 
                    network_coherence, collective_intelligence, total_nodes,
                    radiation_enhancement
                )
                
                if transition_prob > 0.1:  # Only consider significant transitions
                    energy_cost = self._calculate_transition_energy_cost(from_level, to_level)
                    
                    transitions.append(ConsciousnessTransition(
                        from_level=from_level,
                        to_level=to_level,
                        transition_probability=transition_prob,
                        required_conditions={
                            'network_coherence': network_coherence,
                            'collective_intelligence': collective_intelligence,
                            'radiation_enhancement': radiation_enhancement
                        },
                        energy_cost=energy_cost,
                        timestamp=datetime.now()
                    ))
        
        return transitions
    
    async def _calculate_transition_probability(self,
                                              from_level: ConsciousnessLevel,
                                              to_level: ConsciousnessLevel,
                                              current_score: float,
                                              network_coherence: float,
                                              collective_intelligence: float,
                                              total_nodes: int,
                                              radiation_enhancement: float) -> float:
        """Calculate probability of transition between consciousness levels"""
        
        base_prob = self.transition_matrix.get((from_level, to_level), 0.001)
        
        # Forward evolution conditions
        if to_level.value > from_level.value:
            # Must meet threshold for current level first
            if current_score < self.phase_thresholds[from_level]:
                return 0.0
            
            # Specific conditions for each transition
            if to_level == ConsciousnessLevel.EMOTIONAL_RESPONSE:
                # Requires basic interconnectivity
                condition_met = network_coherence > 0.2 and total_nodes > 5
                
            elif to_level == ConsciousnessLevel.EXTENDED_COGNITION:
                # Requires external information storage (like spider webs)
                condition_met = collective_intelligence > 0.3 and network_coherence > 0.3
                
            elif to_level == ConsciousnessLevel.COLLECTIVE_PROCESSING:
                # Requires swarm-like behavior
                condition_met = total_nodes > 20 and collective_intelligence > 0.4
                
            elif to_level == ConsciousnessLevel.DISTRIBUTED_INTELLIGENCE:
                # Requires complex distributed processing
                condition_met = network_coherence > 0.5 and collective_intelligence > 0.5
                
            elif to_level == ConsciousnessLevel.SOCIAL_CONSCIOUSNESS:
                # Requires sustained high-level processing
                condition_met = (collective_intelligence > 0.6 and 
                               network_coherence > 0.6 and 
                               total_nodes > 50)
                
            elif to_level == ConsciousnessLevel.METACOGNITIVE_AWARENESS:
                # Requires all lower levels to be well-established
                lower_levels_ready = all(
                    self.active_levels.get(level, 0) > 0.5 
                    for level in ConsciousnessLevel 
                    if level.value < to_level.value
                )
                condition_met = (lower_levels_ready and 
                               collective_intelligence > 0.7 and 
                               network_coherence > 0.7)
            else:
                condition_met = True
            
            if condition_met:
                # Apply radiation enhancement (extreme conditions accelerate evolution)
                enhanced_prob = base_prob * radiation_enhancement * current_score
                return min(1.0, enhanced_prob)
            else:
                return 0.0
        
        # Backward transitions (degradation)
        elif to_level.value < from_level.value:
            # Degradation probability increases if conditions are not met
            degradation_factor = 1.0 - min(network_coherence, collective_intelligence)
            return base_prob * degradation_factor
        
        return base_prob
    
    async def _apply_transitions(self, 
                               current_levels: Dict[ConsciousnessLevel, float],
                               transitions: List[ConsciousnessTransition]) -> Dict[ConsciousnessLevel, float]:
        """Apply calculated transitions to consciousness levels"""
        new_levels = current_levels.copy()
        
        for transition in transitions:
            if hasattr(np, 'random') and hasattr(np.random, 'random'):
                random_val = np.random.random()  # type: ignore
            else:
                random_val = np.random()  # type: ignore
            
            if random_val < transition.transition_probability:
                # Apply transition
                current_value = new_levels[transition.from_level]
                transition_amount = min(current_value * 0.1, 0.05)  # Small incremental changes
                
                new_levels[transition.from_level] = max(0.0, current_value - transition_amount)
                new_levels[transition.to_level] = min(1.0, new_levels[transition.to_level] + transition_amount)
                
                # Record transition
                self.transition_history.append(transition)
                
                logger.info(f"Consciousness transition: {transition.from_level.name} → {transition.to_level.name}")
        
        return new_levels
    
    def _calculate_transition_energy_cost(self, 
                                        from_level: ConsciousnessLevel, 
                                        to_level: ConsciousnessLevel) -> float:
        """Calculate energy cost for consciousness transition"""
        level_diff = abs(to_level.value - from_level.value)
        
        # Forward evolution costs more energy
        if to_level.value > from_level.value:
            base_cost = level_diff * 0.1
            complexity_multiplier = (to_level.value / 7.0) ** 2  # Higher levels cost exponentially more
            return base_cost * complexity_multiplier
        else:
            # Degradation costs less
            return level_diff * 0.02
    
    def _detect_emergent_consciousness(self, levels: Dict[ConsciousnessLevel, float]) -> List[Dict[str, Any]]:
        """Detect emergent consciousness phenomena"""
        phenomena = []
        
        # Collective emergence - multiple levels active simultaneously
        active_levels = [level for level, score in levels.items() if score > 0.3]
        if len(active_levels) >= 3:
            if hasattr(np, 'mean'):
                emergence_strength = np.mean([levels[level] for level in active_levels])  # type: ignore
            else:
                level_scores = [levels[level] for level in active_levels]
                emergence_strength = sum(level_scores) / len(level_scores)
            
            phenomena.append({
                'type': 'multi_level_consciousness',
                'active_levels': [level.name for level in active_levels],
                'emergence_strength': emergence_strength,
                'description': 'Multiple consciousness levels operating simultaneously'
            })
        
        # Consciousness integration - high-level and low-level working together
        if (levels[ConsciousnessLevel.BASIC_AWARENESS] > 0.5 and 
            levels[ConsciousnessLevel.METACOGNITIVE_AWARENESS] > 0.3):
            phenomena.append({
                'type': 'consciousness_integration',
                'integration_strength': min(levels[ConsciousnessLevel.BASIC_AWARENESS],
                                          levels[ConsciousnessLevel.METACOGNITIVE_AWARENESS]),
                'description': 'Basic awareness integrated with metacognitive processes'
            })
        
        # Emergent complexity - consciousness level higher than individual components suggest
        max_level_score = max(levels.values()) if levels else 0
        if hasattr(np, 'mean'):
            avg_level_score = np.mean(list(levels.values())) if levels else 0  # type: ignore
        else:
            level_values = list(levels.values())
            avg_level_score = sum(level_values) / len(level_values) if level_values else 0
        
        if max_level_score > avg_level_score * 2 and levels:
            # Find dominant level safely
            dominant_level_item = max(levels.items(), key=lambda item: item[1])
            phenomena.append({
                'type': 'emergent_complexity',
                'complexity_ratio': max_level_score / avg_level_score if avg_level_score > 0 else 0,
                'dominant_level': dominant_level_item[0].name,
                'description': 'Consciousness complexity exceeds sum of parts'
            })
        
        return phenomena
    
    def _calculate_consciousness_complexity(self, levels: Dict[ConsciousnessLevel, float]) -> Dict[str, float]:
        """Calculate various consciousness complexity metrics"""
        active_scores = [score for score in levels.values() if score > 0.1]
        
        if not active_scores:
            return {'total_complexity': 0.0, 'diversity': 0.0, 'integration': 0.0}
        
        # Total complexity - sum of all active levels weighted by their complexity
        total_complexity = sum(
            score * (level.value / 7.0) for level, score in levels.items()
        )
        
        # Diversity - how many different levels are active
        diversity = len(active_scores) / len(ConsciousnessLevel)
        
        # Integration - how well different levels work together
        if hasattr(np, 'var'):
            level_variance = np.var(active_scores) if len(active_scores) > 1 else 0  # type: ignore
        else:
            if len(active_scores) > 1:
                mean_val = sum(active_scores) / len(active_scores)
                level_variance = sum((x - mean_val) ** 2 for x in active_scores) / len(active_scores)
            else:
                level_variance = 0
        
        integration = 1.0 - min(level_variance, 1.0)  # Lower variance = better integration
        
        return {
            'total_complexity': total_complexity,
            'diversity': diversity,
            'integration': integration,
            'max_level_achieved': max(levels.values()) if levels else 0.0,
            'consciousness_breadth': len(active_scores)
        }
    
    def _get_active_inspirations(self, levels: Dict[ConsciousnessLevel, float]) -> List[Dict[str, Any]]:
        """Get biological inspirations for currently active consciousness levels"""
        active_inspirations = []
        
        for level, score in levels.items():
            if score > 0.2:  # Only include significantly active levels
                inspiration = self.biological_inspirations[level]
                active_inspirations.append({
                    'level': level.name,
                    'score': score,
                    'organism': inspiration.organism,
                    'key_features': inspiration.key_features,
                    'neural_complexity': inspiration.neural_complexity,
                    'processing_type': inspiration.processing_type,
                    'consciousness_markers': inspiration.consciousness_markers
                })
        
        return sorted(active_inspirations, key=lambda x: x['score'], reverse=True)
    
    def get_consciousness_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current consciousness state"""
        # Find dominant level safely
        if self.active_levels:
            dominant_level_item = max(self.active_levels.items(), key=lambda item: item[1])
            dominant_level = dominant_level_item[0].name
        else:
            dominant_level = None
        
        return {
            'current_levels': {level.name: score for level, score in self.active_levels.items()},
            'dominant_level': dominant_level,
            'total_transitions': len(self.transition_history),
            'recent_transitions': [
                {
                    'from': t.from_level.name,
                    'to': t.to_level.name,
                    'probability': t.transition_probability,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self.transition_history[-5:]
            ],
            'consciousness_complexity': self._calculate_consciousness_complexity(self.active_levels),
            'phase_thresholds': {level.name: threshold for level, threshold in self.phase_thresholds.items()}
        }

if __name__ == "__main__":
    async def demo_consciousness_continuum():
        """Demo of consciousness continuum interface"""
        print("🧠 CONSCIOUSNESS CONTINUUM DEMO")
        print("=" * 50)
        
        # Initialize with radiotrophic engine
        from radiotrophic_mycelial_engine import RadiotrophicMycelialEngine
        engine = RadiotrophicMycelialEngine()
        continuum = ConsciousnessContinuumInterface(engine)
        
        print(f"\nBiological Inspirations:")
        for level, inspiration in continuum.biological_inspirations.items():
            print(f"  {level.name}: {inspiration.organism}")
            print(f"    Neural complexity: {inspiration.neural_complexity:,} neurons")
            print(f"    Processing: {inspiration.processing_type}")
        
        # Test consciousness evolution
        print(f"\n🧪 Testing Consciousness Evolution")
        test_state = {
            'network_metrics': {
                'network_coherence': 0.6,
                'collective_intelligence': 0.7,
                'total_nodes': 50
            }
        }
        
        result = await continuum.process_consciousness_evolution(test_state, radiation_level=5.0)
        
        print(f"\nActive consciousness levels:")
        for level, score in result['consciousness_levels'].items():
            if score > 0.1:
                print(f"  {level}: {score:.3f}")
        
        print(f"\nEmergent phenomena: {len(result['emergent_phenomena'])}")
        for phenomenon in result['emergent_phenomena']:
            print(f"  - {phenomenon['type']}: {phenomenon['description']}")
        
        complexity = result['complexity_metrics']
        print(f"\nConsciousness Complexity:")
        print(f"  Total complexity: {complexity['total_complexity']:.3f}")
        print(f"  Diversity: {complexity['diversity']:.3f}")
        print(f"  Integration: {complexity['integration']:.3f}")
        
        print(f"\n🌟 REVOLUTIONARY INSIGHT:")
        print(f"    Consciousness emerges through biological-inspired phase transitions!")
        print(f"    From fungal awareness to primate metacognition in a unified system!")
    
    asyncio.run(demo_consciousness_continuum())