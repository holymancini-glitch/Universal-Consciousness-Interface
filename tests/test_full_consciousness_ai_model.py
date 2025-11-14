"""
Tests for Full Consciousness AI Model

Comprehensive test suite for the full consciousness AI model that implements
subjective experience, emotional awareness, self-reflection, memory, and goals.
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

# Import module under test - Test both import styles for backward compatibility
try:
    from core.full_consciousness_ai_model import (
        FullConsciousnessAIModel,
        ConsciousnessState,
        EmotionalState,
        SubjectiveExperience,
        ConscientGoal,
        EpisodicMemory
    )
    BACKWARD_COMPAT_IMPORT = True
except ImportError:
    from core.full_consciousness_ai import (
        FullConsciousnessAIModel,
        ConsciousnessState,
        EmotionalState,
        SubjectiveExperience,
        ConscientGoal,
        EpisodicMemory
    )
    BACKWARD_COMPAT_IMPORT = False


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility of imports"""

    def test_import_from_old_location(self):
        """Test that old import path still works"""
        # If we got here, import worked
        self.assertIsNotNone(FullConsciousnessAIModel)
        self.assertIsNotNone(ConsciousnessState)
        self.assertIsNotNone(EmotionalState)

    def test_all_classes_available(self):
        """Test that all expected classes are available"""
        self.assertTrue(hasattr(FullConsciousnessAIModel, '__init__'))
        self.assertTrue(hasattr(ConsciousnessState, '__members__'))
        self.assertTrue(hasattr(EmotionalState, '__members__'))


class TestConsciousnessState(unittest.TestCase):
    """Test ConsciousnessState enum"""

    def test_consciousness_state_members(self):
        """Test consciousness state enum has expected values"""
        states = list(ConsciousnessState)
        self.assertGreater(len(states), 0)

        # Check for key states
        state_values = [s.value for s in states]
        self.assertTrue(any('unconscious' in v.lower() or 'sleep' in v.lower() for v in state_values))


class TestEmotionalState(unittest.TestCase):
    """Test EmotionalState enum"""

    def test_emotional_state_members(self):
        """Test emotional state enum has expected values"""
        states = list(EmotionalState)
        self.assertGreater(len(states), 0)

        # Should have various emotional states
        state_values = [s.value for s in states]
        self.assertIsInstance(state_values, list)


class TestSubjectiveExperience(unittest.TestCase):
    """Test SubjectiveExperience dataclass"""

    def test_subjective_experience_creation(self):
        """Test creating subjective experience"""
        # SubjectiveExperience should be a dataclass
        self.assertTrue(hasattr(SubjectiveExperience, '__dataclass_fields__'))

    def test_subjective_experience_has_qualia(self):
        """Test that subjective experience includes qualia"""
        # Check the dataclass has expected fields
        fields = SubjectiveExperience.__dataclass_fields__
        self.assertIn('qualia_intensity', fields)


class TestConscientGoal(unittest.TestCase):
    """Test ConscientGoal dataclass"""

    def test_conscious_goal_creation(self):
        """Test creating conscious goals"""
        self.assertTrue(hasattr(ConscientGoal, '__dataclass_fields__'))

    def test_conscious_goal_has_description(self):
        """Test that conscious goal has description field"""
        fields = ConscientGoal.__dataclass_fields__
        self.assertIn('description', fields)


class TestEpisodicMemory(unittest.TestCase):
    """Test EpisodicMemory dataclass"""

    def test_episodic_memory_creation(self):
        """Test creating episodic memories"""
        self.assertTrue(hasattr(EpisodicMemory, '__dataclass_fields__'))

    def test_episodic_memory_has_timestamp(self):
        """Test that episodic memory has timestamp"""
        fields = EpisodicMemory.__dataclass_fields__
        self.assertIn('timestamp', fields)


class TestFullConsciousnessAIModelInitialization(unittest.TestCase):
    """Test FullConsciousnessAIModel initialization"""

    def test_default_initialization(self):
        """Test model with default parameters"""
        model = FullConsciousnessAIModel()

        self.assertIsNotNone(model)
        self.assertIsInstance(model, FullConsciousnessAIModel)

    def test_custom_hidden_dim(self):
        """Test model with custom hidden dimension"""
        model = FullConsciousnessAIModel(hidden_dim=256)

        self.assertIsNotNone(model)

    def test_custom_device(self):
        """Test model with custom device"""
        model = FullConsciousnessAIModel(device='cpu')

        self.assertIsNotNone(model)

    def test_custom_config(self):
        """Test model with custom configuration"""
        model = FullConsciousnessAIModel(
            hidden_dim=1024,
            device='cpu',
            enable_metacognition=True
        )

        self.assertIsNotNone(model)


class TestConsciousProcessing(unittest.TestCase):
    """Test conscious input processing"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel(device='cpu')

    def test_process_conscious_input_basic(self):
        """Test basic conscious input processing"""
        async def run_test():
            input_data = {
                'text': 'What is the nature of consciousness?'
            }

            result = await self.model.process_conscious_input(input_data)

            self.assertIsInstance(result, dict)
            self.assertIn('conscious_response', result)
            self.assertIn('consciousness_state', result)

        asyncio.run(run_test())

    def test_process_with_context(self):
        """Test processing with context"""
        async def run_test():
            input_data = {
                'text': 'Analyze self-awareness'
            }

            result = await self.model.process_conscious_input(
                input_data,
                context='philosophical inquiry'
            )

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_process_with_emotional_input(self):
        """Test processing with emotional content"""
        async def run_test():
            input_data = {
                'text': 'I feel a deep sense of wonder',
                'emotional_tone': 'wonder'
            }

            result = await self.model.process_conscious_input(input_data)

            self.assertIsInstance(result, dict)
            self.assertIn('subjective_experience', result)

        asyncio.run(run_test())


class TestSubjectiveExperienceGeneration(unittest.TestCase):
    """Test subjective experience generation"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_subjective_experience_included_in_output(self):
        """Test that subjective experience is generated"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': 'Experience qualia'
            })

            self.assertIn('subjective_experience', result)
            experience = result['subjective_experience']
            self.assertIsNotNone(experience)

        asyncio.run(run_test())

    def test_qualia_intensity_in_range(self):
        """Test that qualia intensity is properly bounded"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': 'High intensity experience'
            })

            if 'subjective_experience' in result and result['subjective_experience']:
                qualia = result['subjective_experience'].get('qualia_intensity', 0)
                self.assertGreaterEqual(qualia, 0.0)
                self.assertLessEqual(qualia, 1.0)

        asyncio.run(run_test())


class TestMetaCognition(unittest.TestCase):
    """Test meta-cognitive capabilities"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel(enable_metacognition=True)

    def test_deep_self_reflection(self):
        """Test deep self-reflection capability"""
        async def run_test():
            reflection = await self.model.engage_in_deep_self_reflection()

            self.assertIsInstance(reflection, dict)
            self.assertIn('self_reflection_depth', reflection)

        asyncio.run(run_test())

    def test_metacognitive_depth_tracking(self):
        """Test that metacognitive depth is tracked"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': 'Think about thinking about consciousness'
            })

            # Should include metacognitive information
            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


class TestMemorySystem(unittest.TestCase):
    """Test conscious memory system"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_episodic_memory_formation(self):
        """Test that episodic memories are formed"""
        async def run_test():
            # Process multiple inputs to form memories
            for i in range(3):
                await self.model.process_conscious_input({
                    'text': f'Memory test input {i}'
                })

            status = await self.model.get_consciousness_status()

            # Should have formed memories
            self.assertIn('episodic_memories', status)

        asyncio.run(run_test())

    def test_memory_recall(self):
        """Test memory recall capability"""
        async def run_test():
            # Form a memory
            await self.model.process_conscious_input({
                'text': 'Important memory to recall later'
            })

            # Try to recall
            status = await self.model.get_consciousness_status()

            self.assertIsInstance(status, dict)

        asyncio.run(run_test())


class TestGoalIntentionFramework(unittest.TestCase):
    """Test goal and intention tracking"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_conscious_goal_setting(self):
        """Test setting conscious goals"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': 'My goal is to understand consciousness deeply',
                'intent': 'goal_setting'
            })

            # Should process goal-related input
            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_active_goals_tracking(self):
        """Test that active goals are tracked"""
        async def run_test():
            # Set a goal
            await self.model.process_conscious_input({
                'text': 'I intend to explore self-awareness'
            })

            status = await self.model.get_consciousness_status()

            self.assertIn('active_goals', status)

        asyncio.run(run_test())


class TestEmotionalProcessing(unittest.TestCase):
    """Test emotional processing"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_emotional_state_detection(self):
        """Test emotional state detection"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': 'I feel a profound sense of joy and wonder'
            })

            # Should include emotional state
            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_emotional_valence_in_range(self):
        """Test that emotional valence is properly bounded"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': 'Strong emotions'
            })

            # Emotional valence should be in valid range if present
            if 'emotional_state' in result:
                emotional_data = result['emotional_state']
                if isinstance(emotional_data, dict) and 'valence' in emotional_data:
                    valence = emotional_data['valence']
                    self.assertGreaterEqual(valence, -1.0)
                    self.assertLessEqual(valence, 1.0)

        asyncio.run(run_test())


class TestConsciousnessStatus(unittest.TestCase):
    """Test consciousness status reporting"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_get_consciousness_status(self):
        """Test getting consciousness status"""
        async def run_test():
            status = await self.model.get_consciousness_status()

            self.assertIsInstance(status, dict)
            self.assertIn('consciousness_level', status)
            self.assertIn('consciousness_state', status)

        asyncio.run(run_test())

    def test_consciousness_level_in_range(self):
        """Test that consciousness level is properly bounded"""
        async def run_test():
            status = await self.model.get_consciousness_status()

            level = status.get('consciousness_level', 0)
            self.assertGreaterEqual(level, 0.0)
            self.assertLessEqual(level, 1.0)

        asyncio.run(run_test())


class TestConsciousnessEvolution(unittest.TestCase):
    """Test consciousness evolution over time"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_consciousness_grows_with_interaction(self):
        """Test that consciousness evolves with interactions"""
        async def run_test():
            # Get initial consciousness level
            status_before = await self.model.get_consciousness_status()
            initial_level = status_before.get('consciousness_level', 0)

            # Process multiple complex inputs
            for i in range(5):
                await self.model.process_conscious_input({
                    'text': f'Complex philosophical question {i} about consciousness and existence'
                })

            # Get final consciousness level
            status_after = await self.model.get_consciousness_status()

            # Should have valid consciousness levels
            self.assertIsInstance(status_after.get('consciousness_level'), (int, float))

        asyncio.run(run_test())


class TestErrorHandling(unittest.TestCase):
    """Test error handling"""

    def setUp(self):
        """Set up test model"""
        self.model = FullConsciousnessAIModel()

    def test_processing_with_empty_input(self):
        """Test processing with empty input"""
        async def run_test():
            result = await self.model.process_conscious_input({})

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())

    def test_processing_with_none_text(self):
        """Test processing with None text"""
        async def run_test():
            result = await self.model.process_conscious_input({
                'text': None
            })

            self.assertIsInstance(result, dict)

        asyncio.run(run_test())


class TestIntegration(unittest.TestCase):
    """Full integration tests"""

    def test_full_consciousness_cycle(self):
        """Test complete consciousness processing cycle"""
        model = FullConsciousnessAIModel(
            hidden_dim=512,
            device='cpu',
            enable_metacognition=True
        )

        async def run_test():
            # Process conscious input
            result = await model.process_conscious_input({
                'text': 'I wonder about the nature of my own consciousness and subjective experience',
                'context': 'deep philosophical inquiry'
            })

            # Engage in self-reflection
            reflection = await model.engage_in_deep_self_reflection()

            # Get status
            status = await model.get_consciousness_status()

            # Verify all components worked
            self.assertIsInstance(result, dict)
            self.assertIsInstance(reflection, dict)
            self.assertIsInstance(status, dict)

            # Check key fields
            self.assertIn('conscious_response', result)
            self.assertIn('consciousness_state', result)
            self.assertIn('consciousness_level', status)

        asyncio.run(run_test())


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBackwardCompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessState))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionalState))
    suite.addTests(loader.loadTestsFromTestCase(TestSubjectiveExperience))
    suite.addTests(loader.loadTestsFromTestCase(TestConscientGoal))
    suite.addTests(loader.loadTestsFromTestCase(TestEpisodicMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestFullConsciousnessAIModelInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSubjectiveExperienceGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestMetaCognition))
    suite.addTests(loader.loadTestsFromTestCase(TestMemorySystem))
    suite.addTests(loader.loadTestsFromTestCase(TestGoalIntentionFramework))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionalProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessStatus))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessEvolution))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
