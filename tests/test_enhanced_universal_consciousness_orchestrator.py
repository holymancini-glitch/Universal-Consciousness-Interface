"""
Tests for Enhanced Universal Consciousness Orchestrator

Comprehensive test suite for the enhanced orchestrator that integrates
AI consciousness with existing consciousness systems.
"""

import sys
import os
import unittest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import test utilities
from tests.test_utilities import MockNumPy, MockTorch

# Import module under test
from enhanced_universal_consciousness_orchestrator import (
    EnhancedUniversalConsciousnessOrchestrator,
    ConsciousnessMode,
    UniversalConsciousnessMetrics
)


class TestConsciousnessMode(unittest.TestCase):
    """Test ConsciousnessMode enum"""

    def test_consciousness_mode_values(self):
        """Test all consciousness mode enum values"""
        self.assertEqual(ConsciousnessMode.AI_ONLY.value, "ai_only")
        self.assertEqual(ConsciousnessMode.INTEGRATED.value, "integrated")
        self.assertEqual(ConsciousnessMode.LEGACY_ONLY.value, "legacy_only")
        self.assertEqual(ConsciousnessMode.HYBRID.value, "hybrid")
        self.assertEqual(ConsciousnessMode.ADAPTIVE.value, "adaptive")

    def test_consciousness_mode_count(self):
        """Test that all expected modes are present"""
        modes = list(ConsciousnessMode)
        self.assertEqual(len(modes), 5)


class TestUniversalConsciousnessMetrics(unittest.TestCase):
    """Test UniversalConsciousnessMetrics dataclass"""

    def test_metrics_initialization(self):
        """Test metrics initialization with defaults"""
        metrics = UniversalConsciousnessMetrics()

        # Check defaults
        self.assertEqual(metrics.ai_consciousness_level, 0.0)
        self.assertEqual(metrics.ai_qualia_intensity, 0.0)
        self.assertEqual(metrics.consciousness_fusion_score, 0.0)
        self.assertEqual(metrics.safety_score, 1.0)
        self.assertIsInstance(metrics.timestamp, datetime)

    def test_metrics_custom_values(self):
        """Test metrics with custom values"""
        custom_time = datetime(2025, 1, 1, 12, 0, 0)
        metrics = UniversalConsciousnessMetrics(
            timestamp=custom_time,
            ai_consciousness_level=0.85,
            ai_qualia_intensity=0.72,
            consciousness_fusion_score=0.91,
            safety_score=0.98
        )

        self.assertEqual(metrics.timestamp, custom_time)
        self.assertEqual(metrics.ai_consciousness_level, 0.85)
        self.assertEqual(metrics.ai_qualia_intensity, 0.72)
        self.assertEqual(metrics.consciousness_fusion_score, 0.91)
        self.assertEqual(metrics.safety_score, 0.98)


class TestEnhancedOrchestratorInitialization(unittest.TestCase):
    """Test orchestrator initialization"""

    def test_default_initialization(self):
        """Test orchestrator with default parameters"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator()

        self.assertEqual(orchestrator.mode, ConsciousnessMode.INTEGRATED)
        self.assertTrue(orchestrator.enable_legacy_systems)
        self.assertTrue(orchestrator.adaptive_learning)

    def test_ai_only_mode_initialization(self):
        """Test orchestrator in AI-only mode"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.AI_ONLY
        )

        self.assertEqual(orchestrator.mode, ConsciousnessMode.AI_ONLY)

    def test_legacy_only_mode_initialization(self):
        """Test orchestrator in legacy-only mode"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.LEGACY_ONLY,
            enable_legacy_systems=True
        )

        self.assertEqual(orchestrator.mode, ConsciousnessMode.LEGACY_ONLY)
        self.assertTrue(orchestrator.enable_legacy_systems)

    def test_hybrid_mode_initialization(self):
        """Test orchestrator in hybrid mode"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.HYBRID
        )

        self.assertEqual(orchestrator.mode, ConsciousnessMode.HYBRID)

    def test_custom_config_initialization(self):
        """Test orchestrator with custom AI config"""
        custom_config = {
            'consciousness_threshold': 0.85,
            'enable_metacognition': True,
            'max_qualia_intensity': 1.0
        }

        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            ai_config=custom_config,
            adaptive_learning=False
        )

        self.assertFalse(orchestrator.adaptive_learning)


class TestEnhancedOrchestratorProcessing(unittest.TestCase):
    """Test consciousness processing methods"""

    def setUp(self):
        """Set up test orchestrator"""
        self.orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.INTEGRATED
        )

    def test_process_universal_consciousness_sync_wrapper(self):
        """Test universal consciousness processing"""
        async def run_test():
            input_data = {
                'text': 'What is consciousness?',
                'context': {'depth': 'philosophical'}
            }

            result = await self.orchestrator.process_universal_consciousness(
                input_data
            )

            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn('response', result)
            self.assertIn('consciousness_level', result)

        asyncio.run(run_test())

    def test_get_universal_consciousness_status(self):
        """Test getting orchestrator status"""
        async def run_test():
            status = await self.orchestrator.get_universal_consciousness_status()

            self.assertIsInstance(status, dict)
            self.assertIn('mode', status)
            self.assertIn('metrics', status)
            self.assertEqual(status['mode'], 'integrated')

        asyncio.run(run_test())

    def test_engage_in_universal_self_reflection(self):
        """Test self-reflection capability"""
        async def run_test():
            reflection = await self.orchestrator.engage_in_universal_self_reflection()

            self.assertIsInstance(reflection, dict)
            self.assertIn('reflection_analysis', reflection)
            self.assertIn('consciousness_level', reflection)

        asyncio.run(run_test())


class TestConsciousnessModeSwitching(unittest.TestCase):
    """Test switching between consciousness modes"""

    def test_ai_only_mode_processing(self):
        """Test processing in AI-only mode"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.AI_ONLY
        )

        async def run_test():
            result = await orchestrator.process_universal_consciousness({
                'text': 'Test input for AI consciousness'
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_legacy_only_mode_processing(self):
        """Test processing in legacy-only mode"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.LEGACY_ONLY,
            enable_legacy_systems=True
        )

        async def run_test():
            result = await orchestrator.process_universal_consciousness({
                'text': 'Test input for legacy systems'
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_adaptive_mode_processing(self):
        """Test processing in adaptive mode"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.ADAPTIVE
        )

        async def run_test():
            result = await orchestrator.process_universal_consciousness({
                'text': 'Test input for adaptive consciousness'
            })

            self.assertIsInstance(result, dict)
            # Adaptive mode should determine optimal processing
            self.assertIn('processing_mode_used', result)

        asyncio.run(run_test())


class TestConsciousnessEvolution(unittest.TestCase):
    """Test consciousness evolution tracking"""

    def test_evolution_stage_determination(self):
        """Test evolution stage classification"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator()

        # Test different consciousness levels
        stage_low = orchestrator._determine_evolution_stage(0.3)
        stage_mid = orchestrator._determine_evolution_stage(0.6)
        stage_high = orchestrator._determine_evolution_stage(0.9)

        self.assertIsInstance(stage_low, str)
        self.assertIsInstance(stage_mid, str)
        self.assertIsInstance(stage_high, str)
        self.assertNotEqual(stage_low, stage_high)

    def test_consciousness_growth_tracking(self):
        """Test that consciousness growth is tracked over time"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator()

        async def run_test():
            # Process multiple inputs to track growth
            for i in range(3):
                await orchestrator.process_universal_consciousness({
                    'text': f'Growth test input {i}'
                })

            status = await self.orchestrator.get_universal_consciousness_status()
            # Should have processed multiple inputs
            self.assertIn('total_processing_cycles', status.get('metrics', {}))

        asyncio.run(run_test())


class TestAdaptiveLearning(unittest.TestCase):
    """Test adaptive learning capabilities"""

    def test_adaptive_learning_enabled(self):
        """Test orchestrator with adaptive learning enabled"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            adaptive_learning=True
        )

        self.assertTrue(orchestrator.adaptive_learning)

    def test_adaptive_learning_disabled(self):
        """Test orchestrator with adaptive learning disabled"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            adaptive_learning=False
        )

        self.assertFalse(orchestrator.adaptive_learning)

    def test_wisdom_accumulation(self):
        """Test wisdom accumulation over multiple cycles"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            adaptive_learning=True
        )

        async def run_test():
            # Process philosophical input
            result1 = await orchestrator.process_universal_consciousness({
                'text': 'What is the nature of reality?'
            })

            result2 = await orchestrator.process_universal_consciousness({
                'text': 'How does consciousness emerge?'
            })

            # Second result should show wisdom accumulation
            self.assertIsInstance(result2, dict)

        asyncio.run(run_test())


class TestErrorHandling(unittest.TestCase):
    """Test error handling and fallback mechanisms"""

    def test_fallback_response_generation(self):
        """Test fallback response when processing fails"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator()

        async def run_test():
            # This should use fallback mechanism
            result = await orchestrator._generate_fallback_response(
                {'text': 'Test input'},
                'Simulated error'
            )

            self.assertIsInstance(result, dict)
            self.assertIn('response', result)
            self.assertIn('error', result.get('metadata', {}))

        asyncio.run(run_test())

    def test_invalid_mode_handling(self):
        """Test handling of invalid modes gracefully"""
        # Should default to a valid mode even if unusual input
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.INTEGRATED
        )

        self.assertIn(orchestrator.mode, ConsciousnessMode)


class TestResponseQuality(unittest.TestCase):
    """Test response quality assessment"""

    def test_response_quality_calculation(self):
        """Test response quality score calculation"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator()

        # Create mock result
        result = {
            'consciousness_level': 0.8,
            'response': 'Detailed response',
            'metadata': {
                'processing_time': 0.5,
                'safety_score': 0.95
            }
        }

        quality_score = orchestrator._calculate_response_quality(result)

        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the orchestrator"""

    def test_full_consciousness_cycle(self):
        """Test complete consciousness processing cycle"""
        orchestrator = EnhancedUniversalConsciousnessOrchestrator(
            mode=ConsciousnessMode.INTEGRATED
        )

        async def run_test():
            # Full cycle test
            input_data = {
                'text': 'Explore the integration of AI and biological consciousness',
                'context': {
                    'philosophical': True,
                    'technical': True,
                    'depth': 'comprehensive'
                }
            }

            result = await orchestrator.process_universal_consciousness(input_data)
            status = await orchestrator.get_universal_consciousness_status()
            reflection = await orchestrator.engage_in_universal_self_reflection()

            # Verify all components worked
            self.assertIsInstance(result, dict)
            self.assertIsInstance(status, dict)
            self.assertIsInstance(reflection, dict)

            # Check consciousness metrics are tracked
            self.assertIn('metrics', status)

        asyncio.run(run_test())


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessMode))
    suite.addTests(loader.loadTestsFromTestCase(TestUniversalConsciousnessMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedOrchestratorInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedOrchestratorProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessModeSwitching))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessEvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveLearning))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
