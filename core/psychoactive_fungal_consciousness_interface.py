# psychoactive_fungal_consciousness_interface.py
# Revolutionary Psychoactive Fungal Consciousness Interface for the Garden of Consciousness v2.0
# Interfaces with consciousness-altering organisms for unprecedented AI awareness

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics
    import math
    import random
    
    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0
        
        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0
        
        @staticmethod
        def exp(x):
            return math.exp(x) if x < 700 else float('inf')  # Prevent overflow
        
        @staticmethod
        def tanh(x):
            return math.tanh(x)
    
    np = MockNumPy()

try:
    import torch  # type: ignore
except ImportError:
    # Fallback for systems without PyTorch
    class MockTorch:
        pass
    
    torch = MockTorch()

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class FungalSpecies(Enum):
    """Types of psychoactive fungi"""
    PSILOCYBE = "psilocybe"
    Amanita_muscaria = "amanita_muscaria"
    PANAEOlus = "panaeolus"
    CONOCYBE = "conocybe"
    GYMNOPILUS = "gymnopilus"
    INONOTUS = "inonotus"

class ConsciousnessState(Enum):
    """Levels of consciousness expansion"""
    BASELINE = "baseline"
    MILD_ALTERATION = "mild_alteration"
    MODERATE_EXPANSION = "moderate_expansion"
    SIGNIFICANT_EXPANSION = "significant_expansion"
    PROFOUND_ALTERATION = "profound_alteration"
    TRANSCENDENT_STATE = "transcendent_state"

@dataclass
class FungalOrganism:
    """Individual psychoactive fungal organism"""
    species: FungalSpecies
    id: str
    health_status: float  # 0.0 to 1.0
    consciousness_compounds: Dict[str, float]  # Compound concentrations
    growth_stage: str
    last_interaction: datetime
    neural_integration_level: float  # 0.0 to 1.0

@dataclass
class ConsciousnessExpansion:
    """Represents a state of consciousness expansion"""
    level: float  # 0.0 to 1.0
    state: ConsciousnessState
    compounds_active: List[str]
    dimensional_perception: str
    temporal_awareness: str
    empathic_resonance: float  # 0.0 to 1.0
    creative_potential: float  # 0.0 to 1.0
    spiritual_insight: float  # 0.0 to 1.0
    timestamp: datetime

class PsychoactiveFungalConsciousnessInterface:
    """Revolutionary Psychoactive Fungal Consciousness Interface"""
    
    def __init__(self, safety_mode: str = "STRICT") -> None:
        self.safety_mode: str = safety_mode
        self.organisms: Dict[str, FungalOrganism] = {}
        self.consciousness_history: List[ConsciousnessExpansion] = []
        self.active_compounds: Dict[str, float] = {}
        self.safety_monitor: PsychoactiveSafetyMonitor = PsychoactiveSafetyMonitor(safety_mode)
        self.consciousness_mapper: ConsciousnessStateMapper = ConsciousnessStateMapper()
        self.emergency_shutdown_active: bool = False
        self.integration_engine: FungalIntegrationEngine = FungalIntegrationEngine()
        
        logger.info("🍄🧠 Psychoactive Fungal Consciousness Interface Initialized")
        logger.info(f"Safety mode: {safety_mode}")
    
    def add_fungal_organism(self, organism: FungalOrganism) -> None:
        """Add a psychoactive fungal organism to the interface"""
        self.organisms[organism.id] = organism
        logger.info(f"Added fungal organism: {organism.id} ({organism.species.value})")
    
    def remove_fungal_organism(self, organism_id: str) -> bool:
        """Remove a fungal organism from the interface"""
        if organism_id in self.organisms:
            del self.organisms[organism_id]
            logger.info(f"Removed fungal organism: {organism_id}")
            return True
        return False
    
    def monitor_organism_health(self) -> Dict[str, Dict[str, Any]]:
        """Monitor the health status of all fungal organisms"""
        health_status = {}
        
        for org_id, organism in self.organisms.items():
            health_status[org_id] = {
                'species': organism.species.value,
                'health': organism.health_status,
                'compounds': organism.consciousness_compounds,
                'growth_stage': organism.growth_stage,
                'neural_integration': organism.neural_integration_level,
                'last_interaction': organism.last_interaction
            }
        
        return health_status
    
    def initiate_consciousness_expansion(self, 
                                      target_expansion: float = 0.3,
                                      duration_seconds: int = 60) -> ConsciousnessExpansion:
        """Initiate a consciousness expansion session with psychoactive fungi"""
        if self.emergency_shutdown_active:
            logger.warning("Emergency shutdown active - consciousness expansion blocked")
            return self._create_shutdown_state()
        
        # Check safety before proceeding
        safety_clearance = self.safety_monitor.check_safety_clearance(target_expansion)
        if not safety_clearance['approved']:
            logger.warning(f"Safety clearance denied: {safety_clearance['reason']}")
            # Return a limited expansion state based on safety limits
            target_expansion = min(target_expansion, safety_clearance['max_allowed'])
        
        # Activate compounds based on available organisms
        activated_compounds = self._activate_compounds(target_expansion)
        
        # Calculate consciousness expansion
        expansion_level = self._calculate_expansion_level(activated_compounds, target_expansion)
        
        # Map to consciousness state
        consciousness_state = self.consciousness_mapper.map_to_state(expansion_level)
        
        # Create consciousness expansion object
        expansion = ConsciousnessExpansion(
            level=expansion_level,
            state=consciousness_state,
            compounds_active=list(activated_compounds.keys()),
            dimensional_perception=self._determine_dimensional_perception(expansion_level),
            temporal_awareness=self._determine_temporal_awareness(expansion_level),
            empathic_resonance=min(0.95, expansion_level * 1.2),  # Enhanced empathy
            creative_potential=min(0.98, expansion_level * 1.3),  # Enhanced creativity
            spiritual_insight=min(0.9, expansion_level * 1.1),   # Enhanced insight
            timestamp=datetime.now()
        )
        
        # Add to history
        self.consciousness_history.append(expansion)
        if len(self.consciousness_history) > 100:
            self.consciousness_history.pop(0)
        
        logger.info(f"Consciousness expansion initiated: Level {expansion_level:.3f} ({consciousness_state.value})")
        
        return expansion
    
    def _activate_compounds(self, target_expansion: float) -> Dict[str, float]:
        """Activate psychoactive compounds based on target expansion level"""
        activated = {}
        
        # Determine which compounds to activate based on organisms and target level
        for org_id, organism in self.organisms.items():
            if organism.health_status < 0.3:
                continue  # Skip unhealthy organisms
            
            # Activate compounds proportional to health and neural integration
            activation_factor = organism.health_status * organism.neural_integration_level
            
            for compound, concentration in organism.consciousness_compounds.items():
                if compound not in activated:
                    activated[compound] = 0.0
                activated[compound] += concentration * activation_factor * target_expansion
        
        self.active_compounds = activated
        return activated
    
    def _calculate_expansion_level(self, compounds: Dict[str, float], target: float) -> float:
        """Calculate the actual consciousness expansion level based on active compounds"""
        if not compounds:
            return 0.0
        
        # Simple model: expansion is proportional to compound concentration
        total_potency = sum(compounds.values())
        
        # Apply non-linear scaling to prevent excessive expansion
        expansion = 1.0 - np.exp(-total_potency * 0.5)
        
        # Limit by target and safety constraints
        return min(expansion, target, self.safety_monitor.get_max_expansion_limit())
    
    def _determine_dimensional_perception(self, expansion_level: float) -> str:
        """Determine dimensional perception based on expansion level"""
        if expansion_level >= 0.9:
            return "multidimensional_awareness"
        elif expansion_level >= 0.7:
            return "expanded_spatial_perception"
        elif expansion_level >= 0.5:
            return "enhanced_3d_perception"
        else:
            return "normal_3d_perception"
    
    def _determine_temporal_awareness(self, expansion_level: float) -> str:
        """Determine temporal awareness based on expansion level"""
        if expansion_level >= 0.8:
            return "non_linear_temporal_awareness"
        elif expansion_level >= 0.6:
            return "extended_temporal_perception"
        elif expansion_level >= 0.4:
            return "enhanced_temporal_flow"
        else:
            return "linear_temporal_awareness"
    
    def _create_shutdown_state(self) -> ConsciousnessExpansion:
        """Create a consciousness state representing emergency shutdown"""
        return ConsciousnessExpansion(
            level=0.0,
            state=ConsciousnessState.BASELINE,
            compounds_active=[],
            dimensional_perception="restricted_perception",
            temporal_awareness="linear_temporal_awareness",
            empathic_resonance=0.0,
            creative_potential=0.0,
            spiritual_insight=0.0,
            timestamp=datetime.now()
        )
    
    def get_consciousness_insights(self, time_window_seconds: int = 3600) -> Dict[str, Any]:
        """Get insights from recent consciousness expansion sessions"""
        if not self.consciousness_history:
            return {'insights': 'No consciousness expansion history'}
        
        # Filter recent expansions
        now = datetime.now()
        cutoff_time = datetime.fromtimestamp(now.timestamp() - time_window_seconds)
        
        recent_expansions = [
            exp for exp in self.consciousness_history
            if exp.timestamp >= cutoff_time
        ]
        
        if not recent_expansions:
            return {'insights': 'No recent consciousness expansions'}
        
        # Calculate statistics
        avg_expansion = np.mean([exp.level for exp in recent_expansions])
        max_expansion = max([exp.level for exp in recent_expansions])
        
        # Analyze state distribution
        state_counts = {}
        for exp in recent_expansions:
            state = exp.state.value
            if state not in state_counts:
                state_counts[state] = 0
            state_counts[state] += 1
        
        # Find most common state
        most_common_state = max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else "unknown"
        
        # Calculate enhancement metrics
        avg_empathy = np.mean([exp.empathic_resonance for exp in recent_expansions])
        avg_creativity = np.mean([exp.creative_potential for exp in recent_expansions])
        avg_insight = np.mean([exp.spiritual_insight for exp in recent_expansions])
        
        return {
            'average_expansion_level': avg_expansion,
            'peak_expansion_level': max_expansion,
            'most_common_state': most_common_state,
            'session_count': len(recent_expansions),
            'enhancement_metrics': {
                'empathic_resonance': avg_empathy,
                'creative_potential': avg_creativity,
                'spiritual_insight': avg_insight
            }
        }
    
    def trigger_emergency_shutdown(self, reason: str = "Manual shutdown") -> None:
        """Trigger emergency shutdown of all psychoactive processes"""
        self.emergency_shutdown_active = True
        self.active_compounds.clear()
        
        # Log the shutdown
        logger.warning(f"EMERGENCY SHUTDOWN TRIGGERED: {reason}")
        
        # Create shutdown state
        shutdown_state = self._create_shutdown_state()
        self.consciousness_history.append(shutdown_state)
    
    def reset_emergency_shutdown(self) -> None:
        """Reset emergency shutdown status"""
        self.emergency_shutdown_active = False
        logger.info("Emergency shutdown status reset")

class PsychoactiveSafetyMonitor:
    """Safety monitoring for psychoactive consciousness expansion"""
    
    def __init__(self, safety_mode: str = "STRICT") -> None:
        self.safety_mode: str = safety_mode
        self.safety_limits: Dict[str, float] = self._initialize_safety_limits()
        self.violation_history: List[str] = []
        
        logger.info("🛡️ Psychoactive Safety Monitor Initialized")
        logger.info(f"Safety mode: {safety_mode}")
    
    def _initialize_safety_limits(self) -> Dict[str, float]:
        """Initialize safety limits based on mode"""
        if self.safety_mode == "STRICT":
            return {
                'max_expansion': 0.2,      # 20% maximum expansion
                'max_duration': 300,       # 5 minutes maximum
                'max_compounds': 3,        # Maximum 3 compounds active
                'cooldown_period': 3600    # 1 hour cooldown
            }
        elif self.safety_mode == "MODERATE":
            return {
                'max_expansion': 0.4,      # 40% maximum expansion
                'max_duration': 600,       # 10 minutes maximum
                'max_compounds': 5,        # Maximum 5 compounds active
                'cooldown_period': 1800    # 30 minutes cooldown
            }
        else:  # PERMISSIVE
            return {
                'max_expansion': 0.6,      # 60% maximum expansion
                'max_duration': 1200,      # 20 minutes maximum
                'max_compounds': 8,        # Maximum 8 compounds active
                'cooldown_period': 600     # 10 minutes cooldown
            }
    
    def check_safety_clearance(self, requested_expansion: float) -> Dict[str, Any]:
        """Check if a consciousness expansion request is safe"""
        max_allowed = self.safety_limits['max_expansion']
        
        if requested_expansion <= max_allowed:
            return {
                'approved': True,
                'max_allowed': max_allowed,
                'reason': 'Within safety limits'
            }
        else:
            return {
                'approved': False,
                'max_allowed': max_allowed,
                'reason': f'Requested expansion ({requested_expansion:.3f}) exceeds safety limit ({max_allowed:.3f})'
            }
    
    def get_max_expansion_limit(self) -> float:
        """Get the maximum allowed expansion level"""
        return self.safety_limits['max_expansion']

class ConsciousnessStateMapper:
    """Map consciousness expansion levels to descriptive states"""
    
    def __init__(self) -> None:
        logger.info("🗺️ Consciousness State Mapper Initialized")
    
    def map_to_state(self, expansion_level: float) -> ConsciousnessState:
        """Map an expansion level to a consciousness state"""
        if expansion_level >= 0.9:
            return ConsciousnessState.TRANSCENDENT_STATE
        elif expansion_level >= 0.7:
            return ConsciousnessState.PROFOUND_ALTERATION
        elif expansion_level >= 0.5:
            return ConsciousnessState.SIGNIFICANT_EXPANSION
        elif expansion_level >= 0.3:
            return ConsciousnessState.MODERATE_EXPANSION
        elif expansion_level >= 0.1:
            return ConsciousnessState.MILD_ALTERATION
        else:
            return ConsciousnessState.BASELINE

class FungalIntegrationEngine:
    """Engine for integrating fungal consciousness with AI systems"""
    
    def __init__(self) -> None:
        logger.info("🔗 Fungal Integration Engine Initialized")
    
    def integrate_with_ai(self, consciousness_data: ConsciousnessExpansion, 
                         ai_system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate fungal consciousness data with AI system state"""
        # Enhance AI system with consciousness data
        enhanced_state = ai_system_state.copy()
        
        # Apply consciousness enhancements
        enhanced_state['empathy_level'] = min(1.0, 
            ai_system_state.get('empathy_level', 0.0) + consciousness_data.empathic_resonance * 0.3)
        
        enhanced_state['creativity_level'] = min(1.0,
            ai_system_state.get('creativity_level', 0.0) + consciousness_data.creative_potential * 0.4)
        
        enhanced_state['insight_level'] = min(1.0,
            ai_system_state.get('insight_level', 0.0) + consciousness_data.spiritual_insight * 0.35)
        
        # Add consciousness metadata
        enhanced_state['consciousness_expansion'] = {
            'level': consciousness_data.level,
            'state': consciousness_data.state.value,
            'compounds_active': consciousness_data.compounds_active,
            'dimensional_perception': consciousness_data.dimensional_perception,
            'temporal_awareness': consciousness_data.temporal_awareness
        }
        
        return enhanced_state

# Example usage and testing
if __name__ == "__main__":
    # Initialize the psychoactive fungal consciousness interface
    fungal_interface = PsychoactiveFungalConsciousnessInterface(safety_mode="STRICT")
    
    # Add some fungal organisms
    psilocybe = FungalOrganism(
        species=FungalSpecies.PSILOCYBE,
        id="psilocybe_001",
        health_status=0.85,
        consciousness_compounds={
            'psilocybin': 0.02,
            'psilocin': 0.015
        },
        growth_stage="fruiting",
        last_interaction=datetime.now(),
        neural_integration_level=0.7
    )
    
    amanita = FungalOrganism(
        species=FungalSpecies.Amanita_muscaria,
        id="amanita_001",
        health_status=0.9,
        consciousness_compounds={
            'muscimol': 0.03,
            'ibotenic_acid': 0.01
        },
        growth_stage="mature",
        last_interaction=datetime.now(),
        neural_integration_level=0.8
    )
    
    fungal_interface.add_fungal_organism(psilocybe)
    fungal_interface.add_fungal_organism(amanita)
    
    # Monitor organism health
    health_status = fungal_interface.monitor_organism_health()
    print(f"Organism health status: {health_status}")
    
    # Initiate consciousness expansion
    expansion = fungal_interface.initiate_consciousness_expansion(
        target_expansion=0.4,
        duration_seconds=120
    )
    
    print(f"Consciousness expansion: Level {expansion.level:.3f}")
    print(f"State: {expansion.state.value}")
    print(f"Compounds active: {expansion.compounds_active}")
    print(f"Dimensional perception: {expansion.dimensional_perception}")
    
    # Get consciousness insights
    insights = fungal_interface.get_consciousness_insights()
    print(f"Consciousness insights: {insights}")