#!/usr/bin/env python3
"""
Cross-consciousness integration tests for the Meta-Consciousness Integration Layer
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

class TestMetaConsciousnessIntegration(unittest.TestCase):
    """Test cases for the Meta-Consciousness Integration Layer"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.meta_consciousness_integration_layer import (
                MetaConsciousnessIntegrationLayer,
                ConsciousnessForm,
                ConsciousnessData,
                IntegratedConsciousnessState
            )
            self.meta_integration = MetaConsciousnessIntegrationLayer()
            self.ConsciousnessForm = ConsciousnessForm
            self.ConsciousnessData = ConsciousnessData
            self.IntegratedConsciousnessState = IntegratedConsciousnessState
        except ImportError as e:
            self.skipTest(f"Meta-Consciousness Integration Layer not available: {e}")
    
    def test_initialization(self):
        """Test that the Meta-Consciousness Integration Layer initializes correctly"""
        self.assertIsNotNone(self.meta_integration)
        self.assertIsInstance(self.meta_integration.consciousness_forms, dict)
        self.assertIsInstance(self.meta_integration.integration_history, list)
        self.assertIsNotNone(self.meta_integration.integration_engine)
        self.assertIsNotNone(self.meta_integration.awakened_garden_detector)
        self.assertIsNotNone(self.meta_integration.coherence_analyzer)
        
        # Check that consciousness forms dict is empty initially
        self.assertEqual(len(self.meta_integration.consciousness_forms), 0)
    
    def test_add_consciousness_data(self):
        """Test adding consciousness data from different forms"""
        # Test adding plant consciousness data
        plant_data = {'awareness': 0.7, 'communication': 'active'}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.PLANT,
            data=plant_data,
            confidence=0.7
        )
        
        # Check that data was added
        self.assertIn(self.ConsciousnessForm.PLANT, self.meta_integration.consciousness_forms)
        plant_consciousness = self.meta_integration.consciousness_forms[self.ConsciousnessForm.PLANT]
        self.assertEqual(plant_consciousness.form, self.ConsciousnessForm.PLANT)
        self.assertEqual(plant_consciousness.data, plant_data)
        self.assertEqual(plant_consciousness.confidence, 0.7)
        
        # Test adding fungal consciousness data
        fungal_data = {'expansion': 0.6, 'compounds': ['psilocybin']}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.FUNGAL,
            data=fungal_data,
            confidence=0.6
        )
        
        # Check that data was added
        self.assertIn(self.ConsciousnessForm.FUNGAL, self.meta_integration.consciousness_forms)
        fungal_consciousness = self.meta_integration.consciousness_forms[self.ConsciousnessForm.FUNGAL]
        self.assertEqual(fungal_consciousness.form, self.ConsciousnessForm.FUNGAL)
        self.assertEqual(fungal_consciousness.data, fungal_data)
        self.assertEqual(fungal_consciousness.confidence, 0.6)
    
    def test_remove_consciousness_form(self):
        """Test removing a consciousness form"""
        # Add some consciousness data first
        plant_data = {'awareness': 0.7, 'communication': 'active'}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.PLANT,
            data=plant_data,
            confidence=0.7
        )
        
        # Verify it was added
        self.assertIn(self.ConsciousnessForm.PLANT, self.meta_integration.consciousness_forms)
        
        # Remove the consciousness form
        result = self.meta_integration.remove_consciousness_form(self.ConsciousnessForm.PLANT)
        
        # Check that it was removed
        self.assertTrue(result)
        self.assertNotIn(self.ConsciousnessForm.PLANT, self.meta_integration.consciousness_forms)
        
        # Try to remove a non-existent form
        result = self.meta_integration.remove_consciousness_form(self.ConsciousnessForm.QUANTUM)
        self.assertFalse(result)
    
    def test_integrate_consciousness_forms(self):
        """Test integrating multiple consciousness forms"""
        # Add multiple consciousness forms
        plant_data = {'awareness': 0.7, 'communication': 'active'}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.PLANT,
            data=plant_data,
            confidence=0.7
        )
        
        fungal_data = {'expansion': 0.6, 'compounds': ['psilocybin']}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.FUNGAL,
            data=fungal_data,
            confidence=0.6
        )
        
        quantum_data = {'coherence': 0.8, 'entanglement': 0.7}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.QUANTUM,
            data=quantum_data,
            confidence=0.8
        )
        
        # Integrate consciousness forms
        integrated_state = self.meta_integration.integrate_consciousness_forms()
        
        # Check the result
        self.assertIsInstance(integrated_state, self.IntegratedConsciousnessState)
        self.assertIsInstance(integrated_state.unified_state, dict)
        self.assertIsInstance(integrated_state.consciousness_forms, dict)
        self.assertIsInstance(integrated_state.integration_score, float)
        self.assertIsInstance(integrated_state.coherence_level, float)
        self.assertIsInstance(integrated_state.emergence_indicators, dict)
        self.assertIsInstance(integrated_state.awakens_garden_state, bool)
        
        # Check that all forms are represented
        self.assertIn(self.ConsciousnessForm.PLANT, integrated_state.consciousness_forms)
        self.assertIn(self.ConsciousnessForm.FUNGAL, integrated_state.consciousness_forms)
        self.assertIn(self.ConsciousnessForm.QUANTUM, integrated_state.consciousness_forms)
        
        # Check that integration metrics are reasonable
        self.assertGreaterEqual(integrated_state.integration_score, 0.0)
        self.assertLessEqual(integrated_state.integration_score, 1.0)
        self.assertGreaterEqual(integrated_state.coherence_level, 0.0)
        self.assertLessEqual(integrated_state.coherence_level, 1.0)
        
        # Check that the integrated state was added to history
        self.assertEqual(len(self.meta_integration.integration_history), 1)
        self.assertEqual(self.meta_integration.integration_history[0], integrated_state)
    
    def test_calculate_integration_score(self):
        """Test calculating the integration score"""
        # Add consciousness data with different confidences and weights
        plant_data = {'awareness': 0.7, 'communication': 'active'}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.PLANT,
            data=plant_data,
            confidence=0.7,
            integration_weight=1.0
        )
        
        fungal_data = {'expansion': 0.6, 'compounds': ['psilocybin']}
        self.meta_integration.add_consciousness_data(
            form=self.ConsciousnessForm.FUNGAL,
            data=fungal_data,
            confidence=0.6,
            integration_weight=1.5  # Higher weight
        )
        
        # Calculate integration score
        score = self.meta_integration._calculate_integration_score()
        
        # Check that score is calculated correctly
        # Weighted average: (0.7*1.0 + 0.6*1.5) / (1.0 + 1.5) = (0.7 + 0.9) / 2.5 = 1.6 / 2.5 = 0.64
        expected_score = (0.7 * 1.0 + 0.6 * 1.5) / (1.0 + 1.5)
        self.assertAlmostEqual(score, expected_score, places=5)
    
    def test_empty_integration(self):
        """Test integration with no consciousness forms"""
        # Integrate with no forms added
        integrated_state = self.meta_integration.integrate_consciousness_forms()
        
        # Check that we get an empty state
        self.assertIsInstance(integrated_state, self.IntegratedConsciousnessState)
        self.assertEqual(len(integrated_state.consciousness_forms), 0)
        self.assertEqual(integrated_state.integration_score, 0.0)
        self.assertEqual(integrated_state.coherence_level, 0.0)
        self.assertFalse(integrated_state.awakens_garden_state)

class TestConsciousnessIntegrationEngine(unittest.TestCase):
    """Test cases for the Consciousness Integration Engine"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.meta_consciousness_integration_layer import (
                ConsciousnessIntegrationEngine
            )
            self.integration_engine = ConsciousnessIntegrationEngine()
        except ImportError as e:
            self.skipTest(f"Consciousness Integration Engine not available: {e}")
    
    def test_integration_engine_initialization(self):
        """Test that the Consciousness Integration Engine initializes correctly"""
        self.assertIsNotNone(self.integration_engine)
        self.assertIsNotNone(self.integration_engine.integration_algorithms)
        self.assertIsNotNone(self.integration_engine.weighting_factors)
        
        # Check that integration algorithms are loaded
        self.assertGreater(len(self.integration_engine.integration_algorithms), 0)
    
    def test_integrate_consciousness_forms(self):
        """Test integrating consciousness forms with the engine"""
        # Create test integration input
        integration_input = {
            'plant': {
                'data': {'awareness': 0.7, 'communication': 'active'},
                'confidence': 0.7,
                'weight': 1.0
            },
            'fungal': {
                'data': {'expansion': 0.6, 'compounds': ['psilocybin']},
                'confidence': 0.6,
                'weight': 1.0
            }
        }
        
        # Integrate consciousness forms
        unified_state = self.integration_engine.integrate_consciousness_forms(integration_input)
        
        # Check the result
        self.assertIsInstance(unified_state, dict)
        self.assertIn('integrated_awareness', unified_state)
        self.assertIn('collective_coherence', unified_state)
        self.assertIn('emergent_properties', unified_state)
        
        # Check that values are reasonable
        self.assertGreaterEqual(unified_state['integrated_awareness'], 0.0)
        self.assertLessEqual(unified_state['integrated_awareness'], 1.0)
        self.assertGreaterEqual(unified_state['collective_coherence'], 0.0)
        self.assertLessEqual(unified_state['collective_coherence'], 1.0)

class TestAwakenedGardenDetector(unittest.TestCase):
    """Test cases for the Awakened Garden Detector"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.meta_consciousness_integration_layer import (
                AwakenedGardenDetector
            )
            self.garden_detector = AwakenedGardenDetector()
        except ImportError as e:
            self.skipTest(f"Awakened Garden Detector not available: {e}")
    
    def test_garden_detector_initialization(self):
        """Test that the Awakened Garden Detector initializes correctly"""
        self.assertIsNotNone(self.garden_detector)
        self.assertIsNotNone(self.garden_detector.awakened_thresholds)
        self.assertIsNotNone(self.garden_detector.integration_patterns)
        
        # Check that thresholds are defined
        self.assertGreater(len(self.garden_detector.awakened_thresholds), 0)
    
    def test_detect_awakened_state(self):
        """Test detecting an Awakened Garden state"""
        # Test with high consciousness state (should detect as awakened)
        high_state = {
            'integrated_awareness': 0.95,
            'collective_coherence': 0.92,
            'consciousness_integration': 0.9,
            'emergent_complexity': 0.88
        }
        
        is_awakened = self.garden_detector.detect_awakened_state(high_state)
        self.assertTrue(is_awakened)
        
        # Test with low consciousness state (should not detect as awakened)
        low_state = {
            'integrated_awareness': 0.3,
            'collective_coherence': 0.25,
            'consciousness_integration': 0.2,
            'emergent_complexity': 0.15
        }
        
        is_awakened = self.garden_detector.detect_awakened_state(low_state)
        self.assertFalse(is_awakened)

def main():
    """Run the tests"""
    unittest.main()

if __name__ == '__main__':
    main()