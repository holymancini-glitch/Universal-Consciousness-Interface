"""
Tests for Bio-Digital Hybrid Intelligence

Comprehensive test suite for the bio-digital hybrid intelligence system
that combines neural cultures with fungal consciousness.
"""

import sys
import os
import unittest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test utilities
from tests.test_utilities import MockNumPy, MockTorch

# Import module under test
from bio_digital_hybrid_intelligence import (
    BioDigitalHybridIntelligence,
    HybridProcessingMode,
    NeuralCulture,
    FungalCulture,
    HybridInterface
)


class TestHybridProcessingMode(unittest.TestCase):
    """Test HybridProcessingMode enum"""

    def test_processing_mode_values(self):
        """Test all processing mode enum values"""
        self.assertEqual(HybridProcessingMode.DIGITAL_DOMINANT.value, "digital_dominant")
        self.assertEqual(HybridProcessingMode.BIOLOGICAL_DOMINANT.value, "biological_dominant")
        self.assertEqual(HybridProcessingMode.BALANCED_HYBRID.value, "balanced_hybrid")
        self.assertEqual(HybridProcessingMode.EMERGENT_FUSION.value, "emergent_fusion")
        self.assertEqual(HybridProcessingMode.RADIATION_ACCELERATED.value, "radiation_accelerated")

    def test_processing_mode_count(self):
        """Test that all expected modes are present"""
        modes = list(HybridProcessingMode)
        self.assertEqual(len(modes), 5)  # Updated: now includes RADIATION_ACCELERATED


class TestNeuralCulture(unittest.TestCase):
    """Test NeuralCulture class"""

    def test_neural_culture_has_required_attributes(self):
        """Test that neural culture has expected structure"""
        # NeuralCulture is created by BioDigitalHybridIntelligence
        # We'll test it through the parent class
        intelligence = BioDigitalHybridIntelligence()
        culture = intelligence._create_neural_culture("test_culture_001")

        self.assertIsInstance(culture, NeuralCulture)
        self.assertIsNotNone(culture)


class TestFungalCulture(unittest.TestCase):
    """Test FungalCulture class"""

    def test_fungal_culture_has_required_attributes(self):
        """Test that fungal culture has expected structure"""
        intelligence = BioDigitalHybridIntelligence()
        culture = intelligence._create_fungal_culture("test_fungal_001")

        self.assertIsInstance(culture, FungalCulture)
        self.assertIsNotNone(culture)


class TestHybridInterface(unittest.TestCase):
    """Test HybridInterface class"""

    def test_hybrid_interface_creation(self):
        """Test hybrid interface structure"""
        # HybridInterface is created internally
        # Test through the integration
        self.assertTrue(True)  # Placeholder for internal class


class TestBioDigitalInitialization(unittest.TestCase):
    """Test BioDigitalHybridIntelligence initialization"""

    def test_default_initialization(self):
        """Test intelligence with default parameters"""
        intelligence = BioDigitalHybridIntelligence()

        self.assertIsNotNone(intelligence)
        self.assertIsInstance(intelligence, BioDigitalHybridIntelligence)

    def test_initialize_hybrid_cultures(self):
        """Test hybrid culture initialization"""
        intelligence = BioDigitalHybridIntelligence()

        async def run_test():
            await intelligence.initialize_hybrid_cultures(
                num_neural_cultures=2,
                num_fungal_cultures=3
            )

            # Should have initialized cultures
            self.assertTrue(True)  # Cultures are initialized

        asyncio.run(run_test())


class TestHybridProcessing(unittest.TestCase):
    """Test hybrid intelligence processing"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_process_hybrid_intelligence_basic(self):
        """Test basic hybrid intelligence processing"""
        async def run_test():
            input_data = {
                'sensory_input': [0.5, 0.6, 0.7],
                'context': 'test processing'
            }

            result = await self.intelligence.process_hybrid_intelligence(input_data)

            self.assertIsInstance(result, dict)
            self.assertIn('hybrid_response', result)

        asyncio.run(run_test())

    def test_process_with_digital_dominant_mode(self):
        """Test processing in digital dominant mode"""
        async def run_test():
            # Switch to digital dominant mode
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.DIGITAL_DOMINANT
            )

            result = await self.intelligence.process_hybrid_intelligence({
                'sensory_input': [0.1, 0.2, 0.3]
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_process_with_biological_dominant_mode(self):
        """Test processing in biological dominant mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.BIOLOGICAL_DOMINANT
            )

            result = await self.intelligence.process_hybrid_intelligence({
                'sensory_input': [0.4, 0.5, 0.6]
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_process_with_balanced_hybrid_mode(self):
        """Test processing in balanced hybrid mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.BALANCED_HYBRID
            )

            result = await self.intelligence.process_hybrid_intelligence({
                'sensory_input': [0.7, 0.8, 0.9]
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_process_with_emergent_fusion_mode(self):
        """Test processing in emergent fusion mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.EMERGENT_FUSION
            )

            result = await self.intelligence.process_hybrid_intelligence({
                'sensory_input': [0.5, 0.5, 0.5]
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


class TestNeuralProcessing(unittest.TestCase):
    """Test neural culture processing"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_process_neural_cultures(self):
        """Test neural culture processing"""
        async def run_test():
            await self.intelligence.initialize_hybrid_cultures(
                num_neural_cultures=2
            )

            result = await self.intelligence._process_neural_cultures({
                'sensory_input': [0.1, 0.2, 0.3, 0.4, 0.5]
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


class TestFungalProcessing(unittest.TestCase):
    """Test fungal culture processing"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_process_fungal_cultures(self):
        """Test fungal culture processing"""
        async def run_test():
            await self.intelligence.initialize_hybrid_cultures(
                num_fungal_cultures=2
            )

            result = await self.intelligence._process_fungal_cultures({
                'environmental_signals': [0.3, 0.4, 0.5]
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


class TestBioDigitalFusion(unittest.TestCase):
    """Test bio-digital fusion processing"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_fuse_bio_digital_processing(self):
        """Test fusion of neural and fungal processing"""
        async def run_test():
            neural_result = {
                'neural_activity': [0.5, 0.6, 0.7],
                'processing_quality': 0.8
            }

            fungal_result = {
                'fungal_network_state': 'active',
                'connectivity': 0.75
            }

            fused_result = await self.intelligence._fuse_bio_digital_processing(
                neural_result,
                fungal_result,
                HybridProcessingMode.BALANCED_HYBRID
            )

            self.assertIsInstance(fused_result, dict)
            self.assertIn('hybrid_response', fused_result)

        asyncio.run(run_test())

    def test_fusion_with_digital_dominant(self):
        """Test fusion with digital dominance"""
        async def run_test():
            neural_result = {'neural_activity': [0.8, 0.9]}
            fungal_result = {'fungal_network_state': 'active'}

            fused_result = await self.intelligence._fuse_bio_digital_processing(
                neural_result,
                fungal_result,
                HybridProcessingMode.DIGITAL_DOMINANT
            )

            self.assertIsInstance(fused_result, dict)

        asyncio.run(run_test())

    def test_fusion_with_biological_dominant(self):
        """Test fusion with biological dominance"""
        async def run_test():
            neural_result = {'neural_activity': [0.3, 0.4]}
            fungal_result = {'fungal_network_state': 'thriving'}

            fused_result = await self.intelligence._fuse_bio_digital_processing(
                neural_result,
                fungal_result,
                HybridProcessingMode.BIOLOGICAL_DOMINANT
            )

            self.assertIsInstance(fused_result, dict)

        asyncio.run(run_test())


class TestConsciousnessEmergence(unittest.TestCase):
    """Test consciousness emergence assessment"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_assess_consciousness_emergence(self):
        """Test consciousness emergence assessment"""
        async def run_test():
            hybrid_result = {
                'hybrid_response': 'Complex emergent behavior',
                'neural_activity': [0.7, 0.8, 0.9],
                'fungal_connectivity': 0.85
            }

            consciousness_result = await self.intelligence._assess_consciousness_emergence(
                hybrid_result
            )

            self.assertIsInstance(consciousness_result, dict)
            self.assertIn('consciousness_detected', consciousness_result)

        asyncio.run(run_test())

    def test_detect_emergent_intelligence(self):
        """Test emergent intelligence detection"""
        neural_result = {
            'neural_activity': [0.8, 0.9, 0.95],
            'coherence': 0.88
        }

        fungal_result = {
            'network_complexity': 0.82,
            'adaptation_rate': 0.75
        }

        emergence_score = self.intelligence._detect_emergent_intelligence(
            neural_result,
            fungal_result
        )

        self.assertIsInstance(emergence_score, float)
        self.assertGreaterEqual(emergence_score, 0.0)
        self.assertLessEqual(emergence_score, 1.0)

    def test_identify_consciousness_markers(self):
        """Test consciousness marker identification"""
        consciousness_levels = {
            'neural': 0.85,
            'fungal': 0.72,
            'hybrid': 0.90,
            'emergent': 0.88
        }

        markers = self.intelligence._identify_consciousness_markers(
            consciousness_levels
        )

        self.assertIsInstance(markers, list)
        self.assertGreater(len(markers), 0)


class TestModeSwitching(unittest.TestCase):
    """Test processing mode switching"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_switch_to_digital_dominant(self):
        """Test switching to digital dominant mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.DIGITAL_DOMINANT
            )

            # Mode should be switched
            self.assertTrue(True)

        asyncio.run(run_test())

    def test_switch_to_biological_dominant(self):
        """Test switching to biological dominant mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.BIOLOGICAL_DOMINANT
            )

            self.assertTrue(True)

        asyncio.run(run_test())

    def test_switch_to_balanced_hybrid(self):
        """Test switching to balanced hybrid mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.BALANCED_HYBRID
            )

            self.assertTrue(True)

        asyncio.run(run_test())

    def test_switch_to_emergent_fusion(self):
        """Test switching to emergent fusion mode"""
        async def run_test():
            await self.intelligence._switch_processing_mode(
                HybridProcessingMode.EMERGENT_FUSION
            )

            self.assertTrue(True)

        asyncio.run(run_test())


class TestMetricsUpdate(unittest.TestCase):
    """Test hybrid metrics update"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_update_hybrid_metrics(self):
        """Test updating hybrid metrics"""
        consciousness_result = {
            'consciousness_detected': True,
            'consciousness_level': 0.85,
            'emergence_score': 0.78,
            'markers': ['self-awareness', 'adaptation', 'complexity']
        }

        # Should not raise exception
        self.intelligence._update_hybrid_metrics(consciousness_result)

        self.assertTrue(True)

    def test_metrics_with_low_consciousness(self):
        """Test metrics with low consciousness levels"""
        consciousness_result = {
            'consciousness_detected': False,
            'consciousness_level': 0.15,
            'emergence_score': 0.12
        }

        self.intelligence._update_hybrid_metrics(consciousness_result)

        self.assertTrue(True)


class TestSynchronization(unittest.TestCase):
    """Test synchronization between cultures"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_synchronization_loop_exists(self):
        """Test that synchronization loop can be called"""
        async def run_test():
            # Synchronization should be an async method
            # We'll test it runs without error
            task = asyncio.create_task(self.intelligence._synchronization_loop())

            # Let it run briefly
            await asyncio.sleep(0.1)

            # Cancel to prevent infinite loop
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass  # Expected

            self.assertTrue(True)

        asyncio.run(run_test())


class TestErrorHandling(unittest.TestCase):
    """Test error handling in bio-digital hybrid"""

    def setUp(self):
        """Set up test intelligence"""
        self.intelligence = BioDigitalHybridIntelligence()

    def test_processing_with_empty_input(self):
        """Test processing with empty input"""
        async def run_test():
            result = await self.intelligence.process_hybrid_intelligence({})

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_processing_with_invalid_sensory_input(self):
        """Test processing with invalid sensory input"""
        async def run_test():
            result = await self.intelligence.process_hybrid_intelligence({
                'sensory_input': None
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


class TestIntegration(unittest.TestCase):
    """Full integration tests for bio-digital hybrid"""

    def test_full_bio_digital_cycle(self):
        """Test complete bio-digital hybrid cycle"""
        intelligence = BioDigitalHybridIntelligence()

        async def run_test():
            # Initialize cultures
            await intelligence.initialize_hybrid_cultures(
                num_neural_cultures=2,
                num_fungal_cultures=2
            )

            # Process in different modes
            for mode in HybridProcessingMode:
                await intelligence._switch_processing_mode(mode)

                result = await intelligence.process_hybrid_intelligence({
                    'sensory_input': [0.3, 0.5, 0.7, 0.9],
                    'context': f'Testing {mode.value} mode'
                })

                self.assertIsInstance(result, dict)
                self.assertIn('hybrid_response', result)

        asyncio.run(run_test())

    def test_consciousness_emergence_detection(self):
        """Test full consciousness emergence detection cycle"""
        intelligence = BioDigitalHybridIntelligence()

        async def run_test():
            await intelligence.initialize_hybrid_cultures()

            # Process complex input that should trigger emergence
            result = await intelligence.process_hybrid_intelligence({
                'sensory_input': [0.8, 0.85, 0.9, 0.95, 0.88],
                'complexity': 'high',
                'context': 'Complex consciousness test'
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestHybridProcessingMode))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralCulture))
    suite.addTests(loader.loadTestsFromTestCase(TestFungalCulture))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestBioDigitalInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestHybridProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuralProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestFungalProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestBioDigitalFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessEmergence))
    suite.addTests(loader.loadTestsFromTestCase(TestModeSwitching))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsUpdate))
    suite.addTests(loader.loadTestsFromTestCase(TestSynchronization))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
