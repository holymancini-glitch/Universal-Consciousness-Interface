# test_integration_modules.py
# Tests for integration modules with Universal Consciousness Interface

import unittest
import numpy as np
import sys
import os
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import integration modules
try:
    from modules.fractal_ai_universal_integration import (
        FractalAIUniversalIntegration, 
        FractalAIMycelialIntegration,
        FractalAIPlantIntegration,
        FractalAIIntegrationState
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

class TestFractalAIUniversalIntegration(unittest.TestCase):
    """Test suite for Fractal AI Universal Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration modules not available")
        
        # Create mock Fractal AI system
        self.mock_fractal_ai = Mock()
        self.mock_fractal_ai.state_history = []
        
        # Create mock Universal Consciousness Orchestrator
        self.mock_universal_orchestrator = Mock()
        
        # Initialize integration
        self.integration = FractalAIUniversalIntegration(
            self.mock_fractal_ai, 
            self.mock_universal_orchestrator
        )
    
    def test_initialization(self):
        """Test integration initialization."""
        self.assertIsNotNone(self.integration.fractal_ai_system)
        self.assertIsNotNone(self.integration.universal_orchestrator)
        self.assertIsInstance(self.integration.integration_history, list)
    
    def test_fractal_ai_state_extraction(self):
        """Test extraction of Fractal AI state."""
        # Mock state history
        mock_state = Mock()
        mock_state.consciousness_level = 0.7
        mock_state.coherence = 0.8
        mock_state.stability = 0.6
        mock_state.integration = 0.5
        mock_state.resonance = True
        mock_state.metrics = {'test_metric': 0.9}
        
        self.mock_fractal_ai.state_history = [mock_state]
        
        fractal_state = self.integration._get_fractal_ai_state()
        
        self.assertEqual(fractal_state['consciousness_level'], 0.7)
        self.assertEqual(fractal_state['coherence'], 0.8)
        self.assertEqual(fractal_state['stability'], 0.6)
        self.assertEqual(fractal_state['integration'], 0.5)
        self.assertTrue(fractal_state['resonance'])
    
    def test_universal_input_preparation(self):
        """Test preparation of input for Universal Consciousness Orchestrator."""
        fractal_state = {
            'consciousness_level': 0.7,
            'coherence': 0.8,
            'stability': 0.6,
            'integration': 0.5,
            'resonance': True
        }
        
        plant_signals = {'frequency': 50, 'amplitude': 0.8}
        environmental_data = {'temperature': 22, 'humidity': 60}
        
        input_vector = self.integration._prepare_universal_input(
            fractal_state, plant_signals, environmental_data
        )
        
        self.assertEqual(input_vector.shape, (256,))
        self.assertEqual(input_vector[0], 0.7)  # consciousness_level
        self.assertEqual(input_vector[1], 0.8)  # coherence
        self.assertEqual(input_vector[2], 0.6)  # stability
        self.assertEqual(input_vector[3], 0.5)  # integration
        self.assertEqual(input_vector[4], 1.0)  # resonance (True -> 1.0)
    
    @patch('modules.fractal_ai_universal_integration.EnhancedMycelialEngine')
    def test_mycelial_integration(self, mock_mycelial_engine):
        """Test integration with Enhanced Mycelial Engine."""
        # Mock mycelial engine
        mock_engine_instance = Mock()
        mock_engine_instance.process_multi_consciousness_input.return_value = {
            'emergent_patterns': ['pattern1', 'pattern2'],
            'network_metrics': {'connectivity': 0.7, 'intelligence': 0.8}
        }
        mock_engine_instance.measure_network_connectivity.return_value = 0.75
        mock_engine_instance.assess_collective_intelligence.return_value = 0.85
        
        mock_mycelial_engine.return_value = mock_engine_instance
        
        # Reinitialize integration with mocked engine
        integration = FractalAIUniversalIntegration(
            self.mock_fractal_ai, 
            self.mock_universal_orchestrator
        )
        
        fractal_state = {
            'consciousness_level': 0.7,
            'coherence': 0.8,
            'stability': 0.6
        }
        
        mock_universal_state = Mock()
        mock_universal_state.unified_consciousness_score = 0.75
        mock_universal_state.quantum_coherence = 0.65
        mock_universal_state.mycelial_connectivity = 0.7
        
        result = integration._process_mycelial_integration(fractal_state, mock_universal_state)
        
        self.assertTrue(result['processed'])
        self.assertEqual(result['connectivity'], 0.75)
        self.assertEqual(result['intelligence'], 0.85)
        self.assertIn('patterns', result)
        self.assertIn('metrics', result)
    
    @patch('modules.fractal_ai_universal_integration.PlantCommunicationInterface')
    def test_plant_integration(self, mock_plant_interface):
        """Test integration with Plant Communication Interface."""
        # Mock plant interface
        mock_interface_instance = Mock()
        mock_interface_instance.decode_electromagnetic_signals.return_value = {'decoded': 'signals'}
        mock_interface_instance.monitor_plant_network.return_value = 0.8
        mock_interface_instance.assess_consciousness_level.return_value = 0.75
        
        mock_plant_interface.return_value = mock_interface_instance
        
        # Reinitialize integration with mocked interface
        integration = FractalAIUniversalIntegration(
            self.mock_fractal_ai, 
            self.mock_universal_orchestrator
        )
        
        plant_signals = {'frequency': 50, 'amplitude': 0.8}
        result = integration._process_plant_integration(plant_signals)
        
        self.assertTrue(result['processed'])
        self.assertIn('decoded_signals', result)
        self.assertEqual(result['network_health'], 0.8)
        self.assertEqual(result['consciousness_level'], 0.75)
    
    def test_integration_coherence_calculation(self):
        """Test calculation of integration coherence."""
        fractal_state = {'coherence': 0.8}
        
        mock_universal_state = Mock()
        mock_universal_state.unified_consciousness_score = 0.7
        
        mycelial_result = {'connectivity': 0.6}
        
        coherence = self.integration._calculate_integration_coherence(
            fractal_state, mock_universal_state, mycelial_result
        )
        
        self.assertIsInstance(coherence, float)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
    
    def test_cross_synchronization_calculation(self):
        """Test calculation of cross-module synchronization."""
        mycelial_result = {'processed': True, 'connectivity': 0.7, 'intelligence': 0.8}
        plant_result = {'processed': True, 'network_health': 0.6, 'consciousness_level': 0.5}
        ecosystem_result = {'processed': True, 'planetary_awareness': 0.4, 'environmental_harmony': 0.3}
        radiotrophic_result = {'processed': True, 'radiation_utilization': 0.9, 'stress_adaptation': 0.8}
        bio_digital_result = {'processed': True, 'hybrid_synchronization': 0.7, 'bio_digital_coherence': 0.6}
        
        synchronization = self.integration._calculate_cross_synchronization(
            mycelial_result, plant_result, ecosystem_result,
            radiotrophic_result, bio_digital_result
        )
        
        self.assertIsInstance(synchronization, float)
        self.assertGreaterEqual(synchronization, 0.0)
        self.assertLessEqual(synchronization, 1.0)

class TestFractalAIMycelialIntegration(unittest.TestCase):
    """Test suite for Fractal AI Mycelial Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration modules not available")
        
        # Create mock Fractal AI system
        self.mock_fractal_ai = Mock()
        self.mock_fractal_ai.state_history = []
        
        # Initialize integration
        self.mycelial_integration = FractalAIMycelialIntegration(self.mock_fractal_ai)
    
    @patch('modules.fractal_ai_universal_integration.EnhancedMycelialEngine')
    def test_fractal_patterns_processing(self, mock_mycelial_engine):
        """Test processing of Fractal AI patterns through mycelial engine."""
        # Mock mycelial engine
        mock_engine_instance = Mock()
        mock_engine_instance.process_multi_consciousness_input.return_value = {
            'processed_layers': {'fractal_ai': {}},
            'emergent_patterns': ['pattern1'],
            'network_metrics': {}
        }
        mock_engine_instance.measure_network_connectivity.return_value = 0.75
        mock_engine_instance.assess_collective_intelligence.return_value = 0.85
        
        mock_mycelial_engine.return_value = mock_engine_instance
        
        # Reinitialize integration with mocked engine
        mycelial_integration = FractalAIMycelialIntegration(self.mock_fractal_ai)
        
        # Mock state history
        mock_state = Mock()
        mock_state.consciousness_level = 0.7
        mock_state.coherence = 0.8
        mock_state.stability = 0.6
        mock_state.integration = 0.5
        mock_state.resonance = True
        
        self.mock_fractal_ai.state_history = [mock_state]
        
        result = mycelial_integration.process_fractal_patterns()
        
        self.assertTrue(result['processed'])
        self.assertIn('mycelial_result', result)
        self.assertEqual(result['network_connectivity'], 0.75)
        self.assertEqual(result['collective_intelligence'], 0.85)

class TestFractalAIPlantIntegration(unittest.TestCase):
    """Test suite for Fractal AI Plant Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration modules not available")
        
        # Create mock Fractal AI system
        self.mock_fractal_ai = Mock()
        self.mock_fractal_ai.state_history = []
        
        # Initialize integration
        self.plant_integration = FractalAIPlantIntegration(self.mock_fractal_ai)
    
    @patch('modules.fractal_ai_universal_integration.PlantCommunicationInterface')
    def test_fractal_to_plant_translation(self, mock_plant_interface):
        """Test translation of Fractal AI output to plant communication signals."""
        # Mock plant interface
        mock_interface_instance = Mock()
        mock_interface_instance.decode_electromagnetic_signals.return_value = {'decoded': 'signals'}
        mock_interface_instance.monitor_plant_network.return_value = 0.8
        mock_interface_instance.assess_consciousness_level.return_value = 0.75
        
        mock_plant_interface.return_value = mock_interface_instance
        
        # Reinitialize integration with mocked interface
        plant_integration = FractalAIPlantIntegration(self.mock_fractal_ai)
        
        # Mock state history
        mock_state = Mock()
        mock_state.consciousness_level = 0.7
        mock_state.coherence = 0.8
        mock_state.integration = 0.5
        mock_state.resonance = True
        
        self.mock_fractal_ai.state_history = [mock_state]
        
        result = plant_integration.translate_fractal_to_plant()
        
        self.assertTrue(result['translated'])
        self.assertIn('plant_signals', result)
        self.assertIn('decoded_signals', result)
        self.assertEqual(result['network_health'], 0.8)
        self.assertEqual(result['plant_consciousness'], 0.75)
        
        # Check that plant signals are correctly formatted
        plant_signals = result['plant_signals']
        self.assertEqual(plant_signals['frequency'], 70)  # 0.7 * 100
        self.assertEqual(plant_signals['amplitude'], 0.8)
        self.assertEqual(plant_signals['pattern_complexity'], 0.5)
        self.assertEqual(plant_signals['resonance'], 1.0)  # True -> 1.0

class TestIntegrationState(unittest.TestCase):
    """Test suite for Integration State data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration modules not available")
    
    def test_integration_state_creation(self):
        """Test creation of integration state."""
        from datetime import datetime
        
        state = FractalAIIntegrationState(
            timestamp=datetime.now(),
            fractal_consciousness_level=0.75,
            integration_coherence=0.8,
            cross_module_synchronization=0.65,
            safety_status="SAFE",
            universal_consciousness_score=0.7
        )
        
        self.assertIsInstance(state.timestamp, datetime)
        self.assertEqual(state.fractal_consciousness_level, 0.75)
        self.assertEqual(state.integration_coherence, 0.8)
        self.assertEqual(state.cross_module_synchronization, 0.65)
        self.assertEqual(state.safety_status, "SAFE")
        self.assertEqual(state.universal_consciousness_score, 0.7)

# Integration tests with mocked Universal Consciousness components
class TestIntegrationWithUniversalConsciousness(unittest.TestCase):
    """Integration tests with mocked Universal Consciousness components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not INTEGRATION_AVAILABLE:
            self.skipTest("Integration modules not available")
        
        # Create mock Fractal AI system
        self.mock_fractal_ai = Mock()
        
        # Create mock Universal Consciousness Orchestrator
        self.mock_universal_orchestrator = Mock()
        
        # Initialize integration
        self.integration = FractalAIUniversalIntegration(
            self.mock_fractal_ai, 
            self.mock_universal_orchestrator
        )
    
    async def test_universal_integration_cycle(self):
        """Test complete integration cycle with Universal Consciousness."""
        # Mock Fractal AI state
        mock_state = Mock()
        mock_state.consciousness_level = 0.7
        mock_state.coherence = 0.8
        mock_state.stability = 0.6
        mock_state.integration = 0.5
        mock_state.resonance = True
        mock_state.metrics = {}
        
        self.mock_fractal_ai.state_history = [mock_state]
        
        # Mock Universal Consciousness response
        mock_consciousness_state = Mock()
        mock_consciousness_state.unified_consciousness_score = 0.75
        mock_consciousness_state.safety_status = "SAFE"
        
        self.mock_universal_orchestrator.consciousness_cycle.return_value = mock_consciousness_state
        
        # Test integration
        integration_state = await self.integration.integrate_with_universal_consciousness(
            plant_signals={'frequency': 50},
            environmental_data={'temperature': 22},
            radiation_data={'level': 0.5}
        )
        
        # Verify results
        self.assertIsInstance(integration_state, FractalAIIntegrationState)
        self.assertEqual(integration_state.fractal_consciousness_level, 0.7)
        self.assertEqual(integration_state.universal_consciousness_score, 0.75)
        self.assertEqual(integration_state.safety_status, "SAFE")
    
    def test_integration_analytics(self):
        """Test integration analytics generation."""
        from datetime import datetime, timedelta
        
        # Create some test history
        for i in range(10):
            state = FractalAIIntegrationState(
                timestamp=datetime.now() - timedelta(minutes=i),
                fractal_consciousness_level=0.5 + 0.1 * i,
                integration_coherence=0.6 + 0.05 * i,
                cross_module_synchronization=0.4 + 0.08 * i,
                safety_status="SAFE" if i < 8 else "WARNING",
                universal_consciousness_score=0.55 + 0.09 * i
            )
            self.integration.integration_history.append(state)
        
        analytics = self.integration.get_integration_analytics()
        
        self.assertIsInstance(analytics, dict)
        self.assertIn('total_integration_cycles', analytics)
        self.assertIn('average_consciousness_level', analytics)
        self.assertIn('average_integration_coherence', analytics)
        self.assertIn('average_synchronization', analytics)
        self.assertIn('safety_status_distribution', analytics)
        self.assertIn('integration_trend', analytics)
        
        # Check values
        self.assertEqual(analytics['total_integration_cycles'], 10)
        self.assertGreater(analytics['average_consciousness_level'], 0)
        self.assertGreater(analytics['average_integration_coherence'], 0)
        self.assertGreater(analytics['average_synchronization'], 0)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)