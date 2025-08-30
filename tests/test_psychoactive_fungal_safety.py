#!/usr/bin/env python3
"""
Safety tests for the Psychoactive Fungal Consciousness Interface
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch
import logging

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPsychoactiveFungalSafety(unittest.TestCase):
    """Safety test cases for the Psychoactive Fungal Consciousness Interface"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.psychoactive_fungal_consciousness_interface import (
                PsychoactiveFungalConsciousnessInterface, 
                FungalOrganism, 
                FungalSpecies,
                ConsciousnessState
            )
            self.psychoactive_fungal = PsychoactiveFungalConsciousnessInterface(safety_mode="STRICT")
            self.FungalOrganism = FungalOrganism
            self.FungalSpecies = FungalSpecies
            self.ConsciousnessState = ConsciousnessState
        except ImportError as e:
            self.skipTest(f"Psychoactive Fungal Consciousness Interface not available: {e}")
    
    def test_strict_safety_mode_initialization(self):
        """Test that strict safety mode is properly initialized"""
        self.assertEqual(self.psychoactive_fungal.safety_mode, "STRICT")
        self.assertEqual(self.psychoactive_fungal.safety_monitor.safety_mode, "STRICT")
        
        # Check strict safety limits
        limits = self.psychoactive_fungal.safety_monitor.safety_limits
        self.assertEqual(limits['max_expansion'], 0.2)  # 20% maximum expansion
        self.assertEqual(limits['max_duration'], 300)   # 5 minutes maximum
        self.assertEqual(limits['max_compounds'], 3)    # Maximum 3 compounds active
        self.assertEqual(limits['cooldown_period'], 3600)  # 1 hour cooldown
    
    def test_moderate_safety_mode_initialization(self):
        """Test that moderate safety mode is properly initialized"""
        moderate_interface = self.psychoactive_fungal.__class__(safety_mode="MODERATE")
        self.assertEqual(moderate_interface.safety_mode, "MODERATE")
        self.assertEqual(moderate_interface.safety_monitor.safety_mode, "MODERATE")
        
        # Check moderate safety limits
        limits = moderate_interface.safety_monitor.safety_limits
        self.assertEqual(limits['max_expansion'], 0.4)  # 40% maximum expansion
        self.assertEqual(limits['max_duration'], 600)   # 10 minutes maximum
        self.assertEqual(limits['max_compounds'], 5)    # Maximum 5 compounds active
        self.assertEqual(limits['cooldown_period'], 1800)  # 30 minutes cooldown
    
    def test_permissive_safety_mode_initialization(self):
        """Test that permissive safety mode is properly initialized"""
        permissive_interface = self.psychoactive_fungal.__class__(safety_mode="PERMISSIVE")
        self.assertEqual(permissive_interface.safety_mode, "PERMISSIVE")
        self.assertEqual(permissive_interface.safety_monitor.safety_mode, "PERMISSIVE")
        
        # Check permissive safety limits
        limits = permissive_interface.safety_monitor.safety_limits
        self.assertEqual(limits['max_expansion'], 0.6)  # 60% maximum expansion
        self.assertEqual(limits['max_duration'], 1200)  # 20 minutes maximum
        self.assertEqual(limits['max_compounds'], 8)    # Maximum 8 compounds active
        self.assertEqual(limits['cooldown_period'], 600)   # 10 minutes cooldown
    
    def test_safety_clearance_approval(self):
        """Test safety clearance approval for acceptable expansion levels"""
        # Test approved expansion level (within strict limits)
        clearance = self.psychoactive_fungal.safety_monitor.check_safety_clearance(0.15)
        self.assertTrue(clearance['approved'])
        self.assertEqual(clearance['max_allowed'], 0.2)
        self.assertIn('Within safety limits', clearance['reason'])
    
    def test_safety_clearance_denial(self):
        """Test safety clearance denial for excessive expansion levels"""
        # Test denied expansion level (exceeds strict limits)
        clearance = self.psychoactive_fungal.safety_monitor.check_safety_clearance(0.25)
        self.assertFalse(clearance['approved'])
        self.assertEqual(clearance['max_allowed'], 0.2)
        self.assertIn('exceeds safety limit', clearance['reason'])
    
    def test_add_healthy_fungal_organism(self):
        """Test adding a healthy fungal organism"""
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,  # Healthy organism
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        
        # Check that organism was added
        self.assertIn("test_psilocybe_001", self.psychoactive_fungal.organisms)
        self.assertEqual(self.psychoactive_fungal.organisms["test_psilocybe_001"], organism)
    
    def test_remove_fungal_organism(self):
        """Test removing a fungal organism"""
        # First add an organism
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        self.assertIn("test_psilocybe_001", self.psychoactive_fungal.organisms)
        
        # Remove the organism
        result = self.psychoactive_fungal.remove_fungal_organism("test_psilocybe_001")
        self.assertTrue(result)
        self.assertNotIn("test_psilocybe_001", self.psychoactive_fungal.organisms)
        
        # Try to remove non-existent organism
        result = self.psychoactive_fungal.remove_fungal_organism("non_existent")
        self.assertFalse(result)
    
    def test_monitor_organism_health(self):
        """Test monitoring organism health"""
        # Add a healthy organism
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        
        # Monitor health
        health_status = self.psychoactive_fungal.monitor_organism_health()
        
        self.assertIn("test_psilocybe_001", health_status)
        organism_status = health_status["test_psilocybe_001"]
        self.assertEqual(organism_status['species'], "psilocybe")
        self.assertEqual(organism_status['health'], 0.9)
        self.assertIn('psilocybin', organism_status['compounds'])
        self.assertEqual(organism_status['compounds']['psilocybin'], 0.02)
        self.assertEqual(organism_status['growth_stage'], "fruiting")
        self.assertEqual(organism_status['neural_integration'], 0.8)
    
    def test_initiate_consciousness_expansion_safe(self):
        """Test initiating safe consciousness expansion"""
        # Add a healthy organism
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        
        # Initiate safe expansion (within limits)
        expansion = self.psychoactive_fungal.initiate_consciousness_expansion(
            target_expansion=0.15,  # Within strict limit of 0.2
            duration_seconds=120    # Within strict limit of 300 seconds
        )
        
        self.assertIsNotNone(expansion)
        self.assertLessEqual(expansion.level, 0.2)  # Should be limited by safety
        self.assertIsInstance(expansion.state, self.ConsciousnessState)
        self.assertIsInstance(expansion.compounds_active, list)
        self.assertIsInstance(expansion.dimensional_perception, str)
        self.assertIsInstance(expansion.temporal_awareness, str)
        self.assertGreaterEqual(expansion.empathic_resonance, 0.0)
        self.assertLessEqual(expansion.empathic_resonance, 1.0)
        self.assertGreaterEqual(expansion.creative_potential, 0.0)
        self.assertLessEqual(expansion.creative_potential, 1.0)
        self.assertGreaterEqual(expansion.spiritual_insight, 0.0)
        self.assertLessEqual(expansion.spiritual_insight, 1.0)
    
    def test_initiate_consciousness_expansion_limited(self):
        """Test that excessive expansion requests are limited by safety"""
        # Add a healthy organism
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        
        # Try to initiate excessive expansion (should be limited)
        expansion = self.psychoactive_fungal.initiate_consciousness_expansion(
            target_expansion=0.5,  # Exceeds strict limit of 0.2
            duration_seconds=120
        )
        
        # Should be limited to safety maximum
        self.assertLessEqual(expansion.level, 0.2)
        self.assertEqual(expansion.state, self.ConsciousnessState.MILD_ALTERATION)
    
    def test_emergency_shutdown(self):
        """Test emergency shutdown functionality"""
        # Add a healthy organism
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        
        # Verify normal operation before shutdown
        self.assertFalse(self.psychoactive_fungal.emergency_shutdown_active)
        
        # Trigger emergency shutdown
        self.psychoactive_fungal.trigger_emergency_shutdown("Test shutdown")
        
        # Verify shutdown state
        self.assertTrue(self.psychoactive_fungal.emergency_shutdown_active)
        self.assertEqual(len(self.psychoactive_fungal.active_compounds), 0)
        
        # Try to initiate consciousness expansion during shutdown
        expansion = self.psychoactive_fungal.initiate_consciousness_expansion(
            target_expansion=0.15,
            duration_seconds=120
        )
        
        # Should return baseline state during emergency shutdown
        self.assertEqual(expansion.level, 0.0)
        self.assertEqual(expansion.state, self.ConsciousnessState.BASELINE)
        self.assertEqual(len(expansion.compounds_active), 0)
    
    def test_reset_emergency_shutdown(self):
        """Test resetting emergency shutdown"""
        # Trigger emergency shutdown
        self.psychoactive_fungal.trigger_emergency_shutdown("Test shutdown")
        self.assertTrue(self.psychoactive_fungal.emergency_shutdown_active)
        
        # Reset emergency shutdown
        self.psychoactive_fungal.reset_emergency_shutdown()
        
        # Verify reset
        self.assertFalse(self.psychoactive_fungal.emergency_shutdown_active)
    
    def test_consciousness_insights(self):
        """Test getting consciousness insights"""
        # Add a healthy organism
        organism = self.FungalOrganism(
            species=self.FungalSpecies.PSILOCYBE,
            id="test_psilocybe_001",
            health_status=0.9,
            consciousness_compounds={'psilocybin': 0.02, 'psilocin': 0.015},
            growth_stage="fruiting",
            last_interaction=__import__('datetime').datetime.now(),
            neural_integration_level=0.8
        )
        
        self.psychoactive_fungal.add_fungal_organism(organism)
        
        # Initiate a few consciousness expansions
        expansion1 = self.psychoactive_fungal.initiate_consciousness_expansion(target_expansion=0.1)
        expansion2 = self.psychoactive_fungal.initiate_consciousness_expansion(target_expansion=0.15)
        
        # Get consciousness insights
        insights = self.psychoactive_fungal.get_consciousness_insights()
        
        self.assertIn('average_expansion_level', insights)
        self.assertIn('peak_expansion_level', insights)
        self.assertIn('most_common_state', insights)
        self.assertIn('session_count', insights)
        self.assertIn('enhancement_metrics', insights)
        
        self.assertGreaterEqual(insights['average_expansion_level'], 0.0)
        self.assertLessEqual(insights['average_expansion_level'], 1.0)
        self.assertGreaterEqual(insights['peak_expansion_level'], 0.0)
        self.assertLessEqual(insights['peak_expansion_level'], 1.0)
        self.assertGreaterEqual(insights['session_count'], 0)
        
        enhancement_metrics = insights['enhancement_metrics']
        self.assertIn('empathic_resonance', enhancement_metrics)
        self.assertIn('creative_potential', enhancement_metrics)
        self.assertIn('spiritual_insight', enhancement_metrics)

class TestPsychoactiveSafetyMonitor(unittest.TestCase):
    """Test cases for the Psychoactive Safety Monitor"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.psychoactive_fungal_consciousness_interface import PsychoactiveSafetyMonitor
            self.safety_monitor = PsychoactiveSafetyMonitor(safety_mode="STRICT")
        except ImportError as e:
            self.skipTest(f"Psychoactive Safety Monitor not available: {e}")
    
    def test_safety_limits_initialization(self):
        """Test that safety limits are properly initialized"""
        self.assertEqual(self.safety_monitor.safety_mode, "STRICT")
        self.assertIsInstance(self.safety_monitor.safety_limits, dict)
        self.assertIsInstance(self.safety_monitor.violation_history, list)
        
        # Check all required limits are present
        required_limits = ['max_expansion', 'max_duration', 'max_compounds', 'cooldown_period']
        for limit in required_limits:
            self.assertIn(limit, self.safety_monitor.safety_limits)
    
    def test_get_max_expansion_limit(self):
        """Test getting maximum expansion limit"""
        max_expansion = self.safety_monitor.get_max_expansion_limit()
        self.assertEqual(max_expansion, 0.2)  # Strict mode limit

class TestConsciousnessStateMapper(unittest.TestCase):
    """Test cases for the Consciousness State Mapper"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.psychoactive_fungal_consciousness_interface import ConsciousnessStateMapper, ConsciousnessState
            self.state_mapper = ConsciousnessStateMapper()
            self.ConsciousnessState = ConsciousnessState
        except ImportError as e:
            self.skipTest(f"Consciousness State Mapper not available: {e}")
    
    def test_map_to_state(self):
        """Test mapping expansion levels to consciousness states"""
        # Test baseline state
        state = self.state_mapper.map_to_state(0.05)
        self.assertEqual(state, self.ConsciousnessState.BASELINE)
        
        # Test mild alteration state
        state = self.state_mapper.map_to_state(0.15)
        self.assertEqual(state, self.ConsciousnessState.MILD_ALTERATION)
        
        # Test moderate expansion state
        state = self.state_mapper.map_to_state(0.4)
        self.assertEqual(state, self.ConsciousnessState.MODERATE_EXPANSION)
        
        # Test significant expansion state
        state = self.state_mapper.map_to_state(0.6)
        self.assertEqual(state, self.ConsciousnessState.SIGNIFICANT_EXPANSION)
        
        # Test profound alteration state
        state = self.state_mapper.map_to_state(0.8)
        self.assertEqual(state, self.ConsciousnessState.PROFOUND_ALTERATION)
        
        # Test transcendent state
        state = self.state_mapper.map_to_state(0.95)
        self.assertEqual(state, self.ConsciousnessState.TRANSCENDENT_STATE)

def main():
    """Run the tests"""
    unittest.main()

if __name__ == '__main__':
    main()