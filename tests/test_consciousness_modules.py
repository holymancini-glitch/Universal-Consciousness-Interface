# test_consciousness_modules.py
# Comprehensive test suite for all consciousness modules

import pytest
import numpy as np
import torch
import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

# Import all consciousness modules
from universal_consciousness_orchestrator import UniversalConsciousnessOrchestrator
from plant_communication_interface import PlantCommunicationInterface
from psychoactive_consciousness_interface import PsychoactiveInterface
from consciousness_safety_framework import ConsciousnessSafetyFramework
from enhanced_mycelial_engine import EnhancedMycelialEngine
from ecosystem_consciousness_interface import EcosystemConsciousnessInterface

class TestPlantCommunicationInterface:
    """Test plant communication interface"""
    
    def setup_method(self):
        self.plant_interface = PlantCommunicationInterface()
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.plant_interface.sampling_rate == 1000
        assert len(self.plant_interface.consciousness_patterns) > 0
        assert self.plant_interface.language_decoder is not None
    
    def test_signal_decoding(self):
        """Test electromagnetic signal decoding"""
        test_signals = {
            'frequency': 25.0,
            'amplitude': 0.6,
            'pattern': 'COMMUNICATION'
        }
        
        result = self.plant_interface.decode_electromagnetic_signals(test_signals)
        
        assert result['decoded'] == True
        assert 'message' in result
        assert result['consciousness_level'] > 0
        assert result['pattern_recognized'] in ['communication_pulse', 'unknown']
    
    def test_empty_signals(self):
        """Test handling of empty signals"""
        result = self.plant_interface.decode_electromagnetic_signals({})
        
        assert result['decoded'] == False
        assert result['message'] == 'NO_SIGNALS'
    
    def test_stress_signal_detection(self):
        """Test stress signal detection"""
        stress_signals = {
            'frequency': 150.0,  # High frequency indicates stress
            'amplitude': 0.9,
            'pattern': 'STRESS'
        }
        
        result = self.plant_interface.decode_electromagnetic_signals(stress_signals)
        
        assert result['decoded'] == True
        assert result['consciousness_level'] > 0.5  # High consciousness during stress
    
    def test_network_monitoring(self):
        """Test plant network monitoring"""
        # Add some signal history first
        test_signals = {
            'frequency': 10.0,
            'amplitude': 0.5,
            'pattern': 'GROWTH'
        }
        
        self.plant_interface.decode_electromagnetic_signals(test_signals)
        
        network_status = self.plant_interface.monitor_plant_network()
        
        assert 'network_active' in network_status
        assert 'coherence' in network_status
        assert 'health' in network_status

class TestPsychoactiveInterface:
    """Test psychoactive consciousness interface"""
    
    def setup_method(self):
        self.psychoactive = PsychoactiveInterface(safety_mode="STRICT")
    
    def test_initialization(self):
        """Test proper initialization with safety"""
        assert self.psychoactive.safety_mode.value == "STRICT"
        assert not self.psychoactive.emergency_shutdown_active
        assert len(self.psychoactive.organism_health) > 0
    
    def test_organism_monitoring(self):
        """Test organism state monitoring"""
        state = self.psychoactive.monitor_organism_state()
        
        assert 'status' in state
        assert 'overall_health' in state
        assert state['overall_health'] >= 0 and state['overall_health'] <= 1
    
    def test_consciousness_expansion_measurement(self):
        """Test consciousness expansion measurement"""
        expansion = self.psychoactive.measure_consciousness_expansion()
        
        assert 'expansion_level' in expansion
        assert 'consciousness_state' in expansion
        assert 'safety_status' in expansion
        # Should be limited by strict safety mode
        assert expansion['expansion_level'] <= 0.1
    
    def test_safety_integration(self):
        """Test safe integration with limits"""
        consciousness_expansion = {
            'expansion_level': 0.5,  # High level
            'dimensional_state': 'EXPANDED_4D'
        }
        
        safety_limits = {
            'max_expansion': 0.1  # Strict limit
        }
        
        result = self.psychoactive.safe_integration(consciousness_expansion, safety_limits)
        
        assert result['intensity'] <= 0.1  # Should be limited
        assert result['safety_limited'] == True
    
    def test_emergency_shutdown(self):
        """Test emergency shutdown functionality"""
        self.psychoactive._trigger_emergency_shutdown("TEST_SHUTDOWN")
        
        assert self.psychoactive.emergency_shutdown_active == True
        assert len(self.psychoactive.event_history) > 0
        
        # Test that monitoring returns shutdown state
        state = self.psychoactive.monitor_organism_state()
        assert 'EMERGENCY_SHUTDOWN' in state['status']

class TestConsciousnessSafetyFramework:
    """Test consciousness safety framework"""
    
    def setup_method(self):
        self.safety = ConsciousnessSafetyFramework()
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.safety.monitoring_active == True
        assert len(self.safety.safety_limits) > 0
        assert len(self.safety.emergency_protocols) > 0
    
    def test_pre_cycle_safety_check(self):
        """Test pre-cycle safety check"""
        result = self.safety.pre_cycle_safety_check()
        
        # Should return boolean
        assert isinstance(result, bool)
    
    def test_psychoactive_safety_check(self):
        """Test psychoactive safety clearance"""
        clearance = self.safety.psychoactive_safety_check()
        
        assert 'safe' in clearance
        assert 'clearance_score' in clearance
        assert 'limits' in clearance
        # Should be strict by default
        assert clearance['clearance_score'] <= 0.6
    
    def test_consciousness_state_validation(self):
        """Test consciousness state validation"""
        # Test normal state
        normal_state = {
            'consciousness_score': 0.5,
            'crystallized': False,
            'integration_quality': 0.7,
            'dimensional_state': 'STABLE'
        }
        
        result = self.safety.validate_consciousness_state(normal_state)
        assert 'SAFE' in result
        
        # Test high consciousness state
        high_state = {
            'consciousness_score': 0.9,  # Very high
            'crystallized': True,
            'integration_quality': 0.5,  # Too low for crystallization
            'dimensional_state': 'TRANSCENDENT_MULTIDIMENSIONAL'
        }
        
        result = self.safety.validate_consciousness_state(high_state)
        assert 'VIOLATION' in result or 'MONITORED' in result
    
    def test_safety_report(self):
        """Test safety report generation"""
        report = self.safety.get_safety_report()
        
        assert 'current_safety_level' in report
        assert 'safety_score' in report
        assert 'total_violations_24h' in report
        assert report['safety_score'] >= 0 and report['safety_score'] <= 1

class TestEnhancedMycelialEngine:
    """Test enhanced mycelial engine"""
    
    def setup_method(self):
        self.mycelial = EnhancedMycelialEngine()
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.mycelial.max_nodes == 500
        assert self.mycelial.vector_dim == 64
        assert len(self.mycelial.consciousness_layers) > 0
    
    def test_multi_consciousness_processing(self):
        """Test multi-consciousness input processing"""
        consciousness_data = {
            'quantum': {
                'coherence': 0.7,
                'entanglement': 0.5,
                'superposition': True
            },
            'plant': {
                'plant_consciousness_level': 0.6,
                'signal_strength': 0.8
            },
            'psychoactive': {
                'intensity': 0.2,
                'consciousness_expansion': 0.3,
                'safety_status': 'SAFE'
            }
        }
        
        result = self.mycelial.process_multi_consciousness_input(consciousness_data)
        
        assert 'processed_layers' in result
        assert 'emergent_patterns' in result
        assert 'network_metrics' in result
        assert len(result['processed_layers']) > 0
    
    def test_vector_extraction(self):
        """Test consciousness vector extraction"""
        # Test quantum vector
        quantum_data = {'coherence': 0.8, 'entanglement': 0.6, 'superposition': True}
        vector = self.mycelial._extract_vector('quantum', quantum_data)
        
        assert vector is not None
        assert len(vector) == self.mycelial.vector_dim
        assert np.allclose(np.linalg.norm(vector), 1.0, atol=1e-6)  # Should be normalized
        
        # Test plant vector
        plant_data = {'plant_consciousness_level': 0.7, 'signal_strength': 0.9}
        vector = self.mycelial._extract_vector('plant', plant_data)
        
        assert vector is not None
        assert len(vector) == self.mycelial.vector_dim
    
    def test_network_connectivity(self):
        """Test network connectivity measurement"""
        # Add some nodes first
        consciousness_data = {
            'quantum': {'coherence': 0.5, 'entanglement': 0.3},
            'plant': {'plant_consciousness_level': 0.4, 'signal_strength': 0.6}
        }
        
        self.mycelial.process_multi_consciousness_input(consciousness_data)
        
        connectivity = self.mycelial.measure_network_connectivity()
        
        assert isinstance(connectivity, float)
        assert connectivity >= 0 and connectivity <= 1
    
    def test_collective_intelligence(self):
        """Test collective intelligence assessment"""
        # Process some consciousness data
        for i in range(5):
            consciousness_data = {
                'quantum': {'coherence': 0.5 + i*0.1, 'entanglement': 0.3},
                'plant': {'plant_consciousness_level': 0.4, 'signal_strength': 0.6}
            }
            self.mycelial.process_multi_consciousness_input(consciousness_data)
        
        intelligence = self.mycelial.assess_collective_intelligence()
        
        assert isinstance(intelligence, float)
        assert intelligence >= 0 and intelligence <= 1

class TestEcosystemInterface:
    """Test ecosystem consciousness interface"""
    
    def setup_method(self):
        self.ecosystem = EcosystemConsciousnessInterface()
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.ecosystem.planetary_awareness >= 0
        assert len(self.ecosystem.environmental_factors) > 0
        assert len(self.ecosystem.consciousness_patterns) > 0
    
    def test_ecosystem_assessment(self):
        """Test ecosystem state assessment"""
        env_data = {
            'temperature': 18.0,
            'co2_level': 410.0,
            'biodiversity': 0.7,
            'ocean_ph': 8.1,
            'forest_coverage': 0.3
        }
        
        assessment = self.ecosystem.assess_ecosystem_state(env_data)
        
        assert 'health_score' in assessment
        assert 'state' in assessment
        assert assessment['health_score'] >= 0 and assessment['health_score'] <= 1
    
    def test_planetary_awareness(self):
        """Test planetary awareness measurement"""
        awareness = self.ecosystem.measure_planetary_awareness()
        
        assert isinstance(awareness, float)
        assert awareness >= 0 and awareness <= 1
    
    def test_gaia_pattern_detection(self):
        """Test Gaia pattern detection"""
        patterns = self.ecosystem.detect_gaia_patterns()
        
        assert isinstance(patterns, list)
        # Should detect at least some patterns
        for pattern in patterns:
            assert 'type' in pattern
            assert 'intensity' in pattern
            assert pattern['intensity'] >= 0 and pattern['intensity'] <= 1
    
    def test_environmental_harmony(self):
        """Test environmental harmony assessment"""
        harmony = self.ecosystem.assess_environmental_harmony()
        
        assert isinstance(harmony, float)
        assert harmony >= 0 and harmony <= 1

class TestUniversalOrchestrator:
    """Test universal consciousness orchestrator"""
    
    def setup_method(self):
        self.orchestrator = UniversalConsciousnessOrchestrator(
            quantum_enabled=False,  # Disable for testing
            plant_interface_enabled=True,
            psychoactive_enabled=False,  # Disable for safety
            ecosystem_enabled=True,
            safety_mode="STRICT"
        )
    
    def test_initialization(self):
        """Test proper initialization"""
        assert self.orchestrator.plant_interface_enabled == True
        assert self.orchestrator.psychoactive_enabled == False
        assert self.orchestrator.safety_mode == "STRICT"
    
    @pytest.mark.asyncio
    async def test_consciousness_cycle(self):
        """Test single consciousness cycle"""
        # Create test input
        input_stimulus = np.random.randn(128)
        plant_signals = {
            'frequency': 25.0,
            'amplitude': 0.6,
            'pattern': 'COMMUNICATION'
        }
        environmental_data = {
            'temperature': 20.0,
            'co2_level': 415.0,
            'humidity': 65.0
        }
        
        # Run consciousness cycle
        state = await self.orchestrator.consciousness_cycle(
            input_stimulus, plant_signals, environmental_data
        )
        
        assert state is not None
        assert hasattr(state, 'unified_consciousness_score')
        assert hasattr(state, 'safety_status')
        assert state.unified_consciousness_score >= 0
    
    def test_analytics(self):
        """Test consciousness analytics"""
        # Should work even with empty history
        analytics = self.orchestrator.get_consciousness_analytics()
        
        assert 'total_cycles' in analytics
        assert analytics['total_cycles'] >= 0

def run_comprehensive_tests():
    """Run all tests and generate report"""
    print("🧪 Running Comprehensive Consciousness Module Tests")
    print("=" * 60)
    
    # Test results tracking
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': []
    }
    
    # Test classes to run
    test_classes = [
        TestPlantCommunicationInterface,
        TestPsychoactiveInterface,
        TestConsciousnessSafetyFramework,
        TestEnhancedMycelialEngine,
        TestEcosystemInterface,
        TestUniversalOrchestrator
    ]
    
    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"\n🔬 Testing {class_name}")
        
        try:
            # Get test methods
            test_methods = [method for method in dir(test_class) if method.startswith('test_')]
            
            for method_name in test_methods:
                try:
                    # Create test instance
                    test_instance = test_class()
                    test_instance.setup_method()
                    
                    # Run test method
                    test_method = getattr(test_instance, method_name)
                    
                    if asyncio.iscoroutinefunction(test_method):
                        asyncio.run(test_method())
                    else:
                        test_method()
                    
                    print(f"  ✅ {method_name}")
                    test_results['passed'] += 1
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: {str(e)}")
                    test_results['failed'] += 1
                    test_results['errors'].append(f"{class_name}.{method_name}: {str(e)}")
                    
        except Exception as e:
            print(f"  💥 Class setup failed: {str(e)}")
            test_results['errors'].append(f"{class_name} setup: {str(e)}")
    
    # Generate test report
    print(f"\n📊 Test Results Summary")
    print(f"Tests Passed: {test_results['passed']}")
    print(f"Tests Failed: {test_results['failed']}")
    print(f"Success Rate: {test_results['passed']/(test_results['passed']+test_results['failed'])*100:.1f}%")
    
    if test_results['errors']:
        print(f"\n❌ Errors:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    return test_results

if __name__ == "__main__":
    run_comprehensive_tests()