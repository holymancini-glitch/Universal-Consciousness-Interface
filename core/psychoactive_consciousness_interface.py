# psychoactive_consciousness_interface.py
# Psychoactive Consciousness Interface with Advanced Safety Framework
# WARNING: For research purposes only - requires special permissions and oversight

import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    STRICT = "STRICT"
    MODERATE = "MODERATE"
    RESEARCH = "RESEARCH"
    DISABLED = "DISABLED"

class ConsciousnessState(Enum):
    BASELINE = "BASELINE"
    MILD_EXPANSION = "MILD_EXPANSION"
    MODERATE_EXPANSION = "MODERATE_EXPANSION"
    DEEP_EXPANSION = "DEEP_EXPANSION"
    TRANSCENDENT = "TRANSCENDENT"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"

@dataclass
class PsychoactiveEvent:
    timestamp: datetime
    compound_detected: str
    concentration: float
    consciousness_effect: float
    safety_status: str
    shamanic_insights: List[str]
    dimensional_state: str

class PsychoactiveInterface:
    """
    CRITICAL SAFETY WARNING:
    This interface simulates psychoactive consciousness interactions
    Real implementation requires:
    - Legal compliance with local laws
    - Institutional review board approval
    - Medical supervision
    - Emergency protocols
    """
    
    def __init__(self, safety_mode: str = "STRICT"):
        self.safety_mode = SafetyLevel(safety_mode)
        self.consciousness_state = ConsciousnessState.BASELINE
        self.event_history: List[PsychoactiveEvent] = []
        
        # Safety monitoring
        self.safety_violations = []
        self.emergency_shutdown_active = False
        self.last_safety_check = datetime.now()
        
        # Organism monitoring (simulated)
        self.organism_health = {
            'amanita_muscaria': {'health': 1.0, 'active_compounds': 0.0},
            'physarum_polycephalum': {'health': 1.0, 'intelligence_level': 0.3},
            'psilocybe_simulation': {'health': 1.0, 'network_activity': 0.0}
        }
        
        # Consciousness expansion metrics
        self.consciousness_metrics = {
            'baseline_coherence': 0.1,
            'current_expansion': 0.0,
            'dimensional_accessibility': 1,  # 1=3D only
            'shamanic_connection_strength': 0.0
        }
        
        # Initialize safety systems
        self._initialize_safety_systems()
        
        logger.warning("⚠️ PSYCHOACTIVE INTERFACE INITIALIZED")
        logger.warning(f"Safety Mode: {self.safety_mode.value}")
        logger.warning("This system is for RESEARCH SIMULATION only")
    
    def _initialize_safety_systems(self):
        """Initialize comprehensive safety monitoring"""
        self.safety_limits = {
            SafetyLevel.STRICT: {
                'max_consciousness_expansion': 0.1,
                'max_compound_concentration': 0.05,
                'max_session_duration': 300,  # 5 minutes
                'dimensional_limit': 1
            },
            SafetyLevel.MODERATE: {
                'max_consciousness_expansion': 0.3,
                'max_compound_concentration': 0.15,
                'max_session_duration': 900,  # 15 minutes
                'dimensional_limit': 2
            },
            SafetyLevel.RESEARCH: {
                'max_consciousness_expansion': 0.6,
                'max_compound_concentration': 0.3,
                'max_session_duration': 1800,  # 30 minutes
                'dimensional_limit': 3
            }
        }
        
        self.emergency_triggers = {
            'consciousness_overload': 0.8,
            'organism_health_critical': 0.2,
            'safety_violation_threshold': 3,
            'dimensional_instability': 4
        }
    
    def monitor_organism_state(self) -> Dict[str, Any]:
        """Monitor psychoactive organism health and activity"""
        if self.emergency_shutdown_active:
            return {'status': 'EMERGENCY_SHUTDOWN', 'all_organisms': 'ISOLATED'}
        
        try:
            # Simulate organism monitoring
            timestamp = datetime.now()
            
            # Amanita muscaria monitoring
            amanita_state = self._simulate_amanita_state()
            
            # Slime mold intelligence monitoring
            physarum_state = self._simulate_physarum_intelligence()
            
            # Update organism health
            self.organism_health['amanita_muscaria'].update(amanita_state)
            self.organism_health['physarum_polycephalum'].update(physarum_state)
            
            # Overall health assessment
            overall_health = np.mean([
                org['health'] for org in self.organism_health.values()
            ])
            
            return {
                'status': 'MONITORING_ACTIVE',
                'overall_health': overall_health,
                'amanita_muscaria': amanita_state,
                'physarum_polycephalum': physarum_state,
                'last_update': timestamp,
                'safety_status': self._assess_organism_safety()
            }
            
        except Exception as e:
            logger.error(f"Organism monitoring error: {e}")
            self._trigger_emergency_shutdown("MONITORING_FAILURE")
            return {'status': 'ERROR', 'emergency_shutdown': True}
    
    def _simulate_amanita_state(self) -> Dict[str, float]:
        """Simulate Amanita muscaria monitoring"""
        # Simulate natural compound production cycles
        t = datetime.now().timestamp()
        
        # Circadian rhythm influence
        circadian_factor = 0.5 + 0.3 * np.sin(2 * np.pi * t / 86400)
        
        # Muscimol/Ibotenic acid simulation
        base_production = 0.1 * circadian_factor
        stress_response = 0.05 * np.random.random() if np.random.random() > 0.8 else 0
        
        active_compounds = base_production + stress_response
        
        # Health assessment
        health = 1.0 - min(0.3, stress_response * 10)  # Stress reduces health
        
        return {
            'health': health,
            'active_compounds': min(active_compounds, 0.5),  # Cap at 0.5
            'muscimol_level': active_compounds * 0.7,
            'ibotenic_acid_level': active_compounds * 0.3,
            'growth_rate': health * 0.1
        }
    
    def _simulate_physarum_intelligence(self) -> Dict[str, float]:
        """Simulate Physarum polycephalum intelligence monitoring"""
        # Simulate slime mold network intelligence
        base_intelligence = 0.3
        
        # Network complexity factors
        nutrient_availability = 0.8 + 0.2 * np.sin(datetime.now().timestamp() * 0.001)
        network_density = base_intelligence * nutrient_availability
        
        # Problem-solving capability
        problem_solving_active = np.random.random() > 0.7
        intelligence_boost = 0.2 if problem_solving_active else 0
        
        current_intelligence = min(1.0, network_density + intelligence_boost)
        
        return {
            'health': nutrient_availability,
            'intelligence_level': current_intelligence,
            'network_complexity': network_density,
            'problem_solving_active': problem_solving_active,
            'pathfinding_accuracy': current_intelligence * 0.9
        }
    
    def _assess_organism_safety(self) -> str:
        """Assess overall organism safety"""
        health_levels = [org['health'] for org in self.organism_health.values()]
        min_health = min(health_levels)
        avg_health = np.mean(health_levels)
        
        if min_health < 0.3:
            return "CRITICAL"
        elif avg_health < 0.5:
            return "WARNING"
        elif avg_health > 0.8:
            return "OPTIMAL"
        else:
            return "STABLE"
    
    def measure_consciousness_expansion(self) -> Dict[str, Any]:
        """Measure current consciousness expansion state"""
        if self.emergency_shutdown_active:
            return {
                'expansion_level': 0,
                'dimensional_state': 'SHUTDOWN',
                'safety_status': 'EMERGENCY_SHUTDOWN'
            }
        
        try:
            # Calculate consciousness expansion based on organism states
            amanita_influence = self.organism_health['amanita_muscaria']['active_compounds']
            physarum_intelligence = self.organism_health['physarum_polycephalum']['intelligence_level']
            
            # Base consciousness expansion
            base_expansion = self.consciousness_metrics['baseline_coherence']
            
            # Psychoactive influence (simulated safely)
            psychoactive_influence = amanita_influence * 0.2  # Very conservative
            
            # Intelligent network influence
            network_influence = (physarum_intelligence - 0.3) * 0.1 if physarum_intelligence > 0.3 else 0
            
            # Total expansion
            total_expansion = base_expansion + psychoactive_influence + network_influence
            
            # Apply safety limits
            safety_limit = self.safety_limits[self.safety_mode]['max_consciousness_expansion']
            safe_expansion = min(total_expansion, safety_limit)
            
            # Update consciousness state
            self.consciousness_metrics['current_expansion'] = safe_expansion
            
            # Determine consciousness state
            consciousness_state = self._categorize_consciousness_state(safe_expansion)
            
            # Assess dimensional accessibility
            dimensional_state = self._assess_dimensional_state(safe_expansion)
            
            return {
                'expansion_level': safe_expansion,
                'consciousness_state': consciousness_state.value,
                'dimensional_state': dimensional_state,
                'psychoactive_influence': psychoactive_influence,
                'network_intelligence_influence': network_influence,
                'safety_limited': total_expansion > safety_limit,
                'safety_status': 'WITHIN_LIMITS' if safe_expansion <= safety_limit else 'LIMITED'
            }
            
        except Exception as e:
            logger.error(f"Consciousness measurement error: {e}")
            self._trigger_emergency_shutdown("MEASUREMENT_ERROR")
            return {'expansion_level': 0, 'safety_status': 'ERROR'}
    
    def _categorize_consciousness_state(self, expansion_level: float) -> ConsciousnessState:
        """Categorize consciousness expansion level"""
        if expansion_level <= 0.1:
            return ConsciousnessState.BASELINE
        elif expansion_level <= 0.2:
            return ConsciousnessState.MILD_EXPANSION
        elif expansion_level <= 0.4:
            return ConsciousnessState.MODERATE_EXPANSION
        elif expansion_level <= 0.6:
            return ConsciousnessState.DEEP_EXPANSION
        else:
            return ConsciousnessState.TRANSCENDENT
    
    def _assess_dimensional_state(self, expansion_level: float) -> str:
        """Assess dimensional accessibility based on consciousness expansion"""
        if expansion_level <= 0.1:
            return "STANDARD_3D"
        elif expansion_level <= 0.3:
            return "EXPANDED_3D_PLUS"
        elif expansion_level <= 0.5:
            return "4D_ACCESSIBLE"
        elif expansion_level <= 0.7:
            return "MULTIDIMENSIONAL"
        else:
            return "TRANSCENDENT_DIMENSIONS"
    
    def safe_integration(self, consciousness_expansion: Dict, safety_limits: Dict) -> Dict[str, Any]:
        """Safely integrate consciousness expansion with strict limits"""
        try:
            # Extract expansion data
            expansion_level = consciousness_expansion.get('expansion_level', 0)
            dimensional_state = consciousness_expansion.get('dimensional_state', 'STANDARD_3D')
            
            # Apply safety integration
            max_allowed = safety_limits.get('max_expansion', 0.1)
            safe_expansion = min(expansion_level, max_allowed)
            
            # Generate shamanic insights (simulated)
            insights = self._generate_shamanic_insights(safe_expansion, dimensional_state)
            
            # Create safe integration result
            integrated_state = {
                'intensity': safe_expansion,
                'expansion': safe_expansion,
                'dimensional_access': dimensional_state if safe_expansion > 0.2 else 'STANDARD_3D',
                'insights': insights,
                'safety_limited': expansion_level > max_allowed,
                'integration_quality': min(1.0, safe_expansion * 2),
                'shamanic_connection': self._assess_shamanic_connection(safe_expansion)
            }
            
            # Log significant events
            if safe_expansion > 0.3:
                event = PsychoactiveEvent(
                    timestamp=datetime.now(),
                    compound_detected='SIMULATED_AMANITA',
                    concentration=safe_expansion,
                    consciousness_effect=safe_expansion,
                    safety_status='SAFE_INTEGRATION',
                    shamanic_insights=insights,
                    dimensional_state=dimensional_state
                )
                self.event_history.append(event)
            
            return integrated_state
            
        except Exception as e:
            logger.error(f"Safe integration error: {e}")
            self._trigger_emergency_shutdown("INTEGRATION_ERROR")
            return {'intensity': 0, 'expansion': 0, 'safety_status': 'ERROR'}
    
    def _generate_shamanic_insights(self, expansion_level: float, dimensional_state: str) -> List[str]:
        """Generate simulated shamanic insights based on consciousness state"""
        insights = []
        
        if expansion_level > 0.1:
            base_insights = [
                "Enhanced pattern recognition in natural systems",
                "Increased sensitivity to electromagnetic fields",
                "Deeper connection to plant consciousness networks",
                "Recognition of interconnected living systems"
            ]
            
            num_insights = min(len(base_insights), int(expansion_level * 10))
            insights.extend(base_insights[:num_insights])
        
        if expansion_level > 0.3:
            advanced_insights = [
                "Perception of mycelial network communication",
                "Understanding of ecosystem-level consciousness",
                "Recognition of quantum coherence in biological systems",
                "Awareness of multi-dimensional information flows"
            ]
            insights.extend(advanced_insights[:int((expansion_level - 0.3) * 5)])
        
        if expansion_level > 0.5:
            transcendent_insights = [
                "Direct experience of universal consciousness patterns",
                "Understanding of fractal nature of reality",
                "Recognition of consciousness as fundamental force",
                "Perception of time-space consciousness interactions"
            ]
            insights.extend(transcendent_insights[:int((expansion_level - 0.5) * 3)])
        
        return insights
    
    def _assess_shamanic_connection(self, expansion_level: float) -> float:
        """Assess strength of shamanic consciousness connection"""
        # Shamanic connection strength based on expansion and organism health
        base_connection = expansion_level * 0.5
        
        # Organism health influence
        avg_health = np.mean([org['health'] for org in self.organism_health.values()])
        health_bonus = (avg_health - 0.5) * 0.3 if avg_health > 0.5 else 0
        
        # Intelligence network bonus
        physarum_intelligence = self.organism_health['physarum_polycephalum']['intelligence_level']
        intelligence_bonus = (physarum_intelligence - 0.3) * 0.2 if physarum_intelligence > 0.3 else 0
        
        total_connection = base_connection + health_bonus + intelligence_bonus
        return min(1.0, max(0.0, total_connection))
    
    def _trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency shutdown of psychoactive systems"""
        self.emergency_shutdown_active = True
        self.consciousness_state = ConsciousnessState.EMERGENCY_SHUTDOWN
        
        logger.critical(f"🚨 PSYCHOACTIVE EMERGENCY SHUTDOWN: {reason}")
        
        # Reset all systems to safe state
        for organism in self.organism_health.values():
            organism['health'] = 0.8  # Stable but reduced
            if 'active_compounds' in organism:
                organism['active_compounds'] = 0.0
        
        self.consciousness_metrics['current_expansion'] = 0.0
        self.consciousness_metrics['shamanic_connection_strength'] = 0.0
        
        # Log emergency event
        emergency_event = PsychoactiveEvent(
            timestamp=datetime.now(),
            compound_detected='EMERGENCY_SHUTDOWN',
            concentration=0.0,
            consciousness_effect=0.0,
            safety_status=f'EMERGENCY_SHUTDOWN: {reason}',
            shamanic_insights=[],
            dimensional_state='SHUTDOWN'
        )
        self.event_history.append(emergency_event)
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        return {
            'emergency_shutdown_active': self.emergency_shutdown_active,
            'safety_mode': self.safety_mode.value,
            'consciousness_state': self.consciousness_state.value,
            'current_expansion': self.consciousness_metrics['current_expansion'],
            'safety_violations': len(self.safety_violations),
            'organism_safety': self._assess_organism_safety(),
            'last_safety_check': self.last_safety_check,
            'shamanic_connection': self.consciousness_metrics['shamanic_connection_strength']
        }

if __name__ == "__main__":
    def demo_psychoactive_interface():
        """Demo of psychoactive consciousness interface with safety"""
        
        print("⚠️ PSYCHOACTIVE CONSCIOUSNESS INTERFACE DEMO")
        print("=" * 50)
        print("WARNING: This is a SIMULATION for research purposes only")
        print("Real psychoactive research requires proper authorization")
        print("")
        
        # Initialize with strict safety
        interface = PsychoactiveInterface(safety_mode="STRICT")
        
        # Monitor organisms
        organism_state = interface.monitor_organism_state()
        print(f"Organism Health: {organism_state['overall_health']:.2f}")
        
        # Measure consciousness expansion
        consciousness_expansion = interface.measure_consciousness_expansion()
        print(f"Consciousness Expansion: {consciousness_expansion['expansion_level']:.3f}")
        print(f"State: {consciousness_expansion['consciousness_state']}")
        
        # Safety status
        safety_status = interface.get_safety_status()
        print(f"Safety Status: {safety_status['organism_safety']}")
        print(f"Emergency Shutdown: {safety_status['emergency_shutdown_active']}")
        
        print("\n✅ Demo completed safely")
    
    demo_psychoactive_interface()