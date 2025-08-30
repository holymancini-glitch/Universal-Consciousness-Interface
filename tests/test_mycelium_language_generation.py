#!/usr/bin/env python3
"""
Language generation tests for the enhanced Mycelium Language Generation System
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

class TestMyceliumLanguageGeneration(unittest.TestCase):
    """Test cases for the Mycelium Language Generation System"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.mycelium_language_generator import (
                MyceliumLanguageGenerator, 
                MyceliumSignal, 
                MyceliumCommunicationType,
                MyceliumWord,
                MyceliumSentence
            )
            self.mycelium_generator = MyceliumLanguageGenerator(network_size=500)
            self.MyceliumCommunicationType = MyceliumCommunicationType
            self.MyceliumSignal = MyceliumSignal
        except ImportError as e:
            self.skipTest(f"Mycelium Language Generator not available: {e}")
    
    def test_initialization(self):
        """Test that the Mycelium Language Generator initializes correctly"""
        self.assertIsNotNone(self.mycelium_generator)
        self.assertEqual(self.mycelium_generator.network_size, 500)
        self.assertIsInstance(self.mycelium_generator.signal_types, list)
        self.assertGreater(len(self.mycelium_generator.signal_types), 0)
        self.assertIsInstance(self.mycelium_generator.phonetic_library, dict)
        self.assertIsInstance(self.mycelium_generator.chemical_vocabulary, dict)
        self.assertIsInstance(self.mycelium_generator.syntactic_rules, dict)
        
        # Check that communication patterns are loaded
        self.assertGreater(len(self.mycelium_generator.communication_patterns), 0)
        
        # Check that language elements are initialized
        self.assertIsInstance(self.mycelium_generator.mycelium_words, list)
        self.assertIsInstance(self.mycelium_generator.mycelium_sentences, list)
    
    def test_signal_generation(self):
        """Test generating mycelium signals"""
        from datetime import datetime
        
        # Create a test signal
        signal = self.MyceliumSignal(
            signal_type=self.MyceliumCommunicationType.CHEMICAL_GRADIENT,
            intensity=0.75,
            duration=2.5,
            spatial_pattern="radial",
            chemical_composition={'glucose': 0.6, 'oxygen': 0.8},
            electrical_frequency=40.0,
            timestamp=datetime.now(),
            network_location=(1.0, 2.0, 0.5)
        )
        
        self.assertEqual(signal.signal_type, self.MyceliumCommunicationType.CHEMICAL_GRADIENT)
        self.assertEqual(signal.intensity, 0.75)
        self.assertEqual(signal.duration, 2.5)
        self.assertEqual(signal.spatial_pattern, "radial")
        self.assertEqual(signal.chemical_composition['glucose'], 0.6)
        self.assertEqual(signal.chemical_composition['oxygen'], 0.8)
        self.assertEqual(signal.electrical_frequency, 40.0)
        self.assertEqual(signal.network_location, (1.0, 2.0, 0.5))
    
    def test_process_mycelium_signal(self):
        """Test processing a mycelium signal"""
        from datetime import datetime
        
        # Create a test signal
        signal = self.MyceliumSignal(
            signal_type=self.MyceliumCommunicationType.ELECTRICAL_PULSE,
            intensity=0.65,
            duration=1.8,
            spatial_pattern="linear",
            chemical_composition={'potassium': 0.4},
            electrical_frequency=25.0,
            timestamp=datetime.now(),
            network_location=(0.5, 1.0, 0.2)
        )
        
        # Process the signal
        processed_signal = self.mycelium_generator.process_mycelium_signal(signal)
        
        # Check the result
        self.assertIsNotNone(processed_signal)
        self.assertEqual(processed_signal.signal_type, signal.signal_type)
        self.assertEqual(processed_signal.intensity, signal.intensity)
        self.assertEqual(processed_signal.duration, signal.duration)
    
    def test_generate_language_from_signals(self):
        """Test generating language from mycelium signals"""
        from datetime import datetime
        
        # Create test signals
        signals = [
            self.MyceliumSignal(
                signal_type=self.MyceliumCommunicationType.CHEMICAL_GRADIENT,
                intensity=0.7,
                duration=2.0,
                spatial_pattern="radial",
                chemical_composition={'glucose': 0.6},
                electrical_frequency=30.0,
                timestamp=datetime.now(),
                network_location=(0.0, 0.0, 0.0)
            ),
            self.MyceliumSignal(
                signal_type=self.MyceliumCommunicationType.ELECTRICAL_PULSE,
                intensity=0.8,
                duration=1.5,
                spatial_pattern="network_wide",
                chemical_composition={},
                electrical_frequency=50.0,
                timestamp=datetime.now(),
                network_location=(1.0, 1.0, 1.0)
            )
        ]
        
        # Generate language from signals
        language_result = self.mycelium_generator.generate_language_from_signals(signals)
        
        # Check the result
        self.assertIsInstance(language_result, dict)
        self.assertIn('words_generated', language_result)
        self.assertIn('sentences_generated', language_result)
        self.assertIn('linguistic_complexity', language_result)
        self.assertIn('semantic_coherence', language_result)
        self.assertIn('generated_words', language_result)
        self.assertIn('generated_sentences', language_result)
        
        # Check that words and sentences were generated
        self.assertGreater(language_result['words_generated'], 0)
        self.assertGreater(language_result['sentences_generated'], 0)
        self.assertGreater(len(language_result['generated_words']), 0)
        self.assertGreater(len(language_result['generated_sentences']), 0)
    
    def test_generate_mycelium_word(self):
        """Test generating a mycelium word"""
        from datetime import datetime
        
        # Create a test signal
        signal = self.MyceliumSignal(
            signal_type=self.MyceliumCommunicationType.CHEMICAL_GRADIENT,
            intensity=0.75,
            duration=2.5,
            spatial_pattern="radial",
            chemical_composition={'glucose': 0.6, 'oxygen': 0.8},
            electrical_frequency=40.0,
            timestamp=datetime.now(),
            network_location=(1.0, 2.0, 0.5)
        )
        
        # Generate a word from the signal
        word = self.mycelium_generator._generate_mycelium_word(signal)
        
        # Check the result
        self.assertIsInstance(word, self.mycelium_generator.MyceliumWord)
        self.assertIsInstance(word.phonetic_pattern, str)
        self.assertIsInstance(word.chemical_signature, dict)
        self.assertIsInstance(word.electrical_signature, float)
        self.assertIsInstance(word.meaning_concept, str)
        self.assertIsInstance(word.context_cluster, str)
        self.assertEqual(word.formation_signals[0], signal)
        
        # Check that the word has meaningful content
        self.assertGreater(len(word.phonetic_pattern), 0)
        self.assertGreater(len(word.meaning_concept), 0)
        self.assertGreater(len(word.context_cluster), 0)
    
    def test_generate_mycelium_sentence(self):
        """Test generating a mycelium sentence"""
        from datetime import datetime
        
        # Create test signals
        signals = [
            self.MyceliumSignal(
                signal_type=self.MyceliumCommunicationType.CHEMICAL_GRADIENT,
                intensity=0.7,
                duration=2.0,
                spatial_pattern="radial",
                chemical_composition={'glucose': 0.6},
                electrical_frequency=30.0,
                timestamp=datetime.now(),
                network_location=(0.0, 0.0, 0.0)
            )
        ]
        
        # Generate words from signals
        words = [self.mycelium_generator._generate_mycelium_word(signal) for signal in signals]
        
        # Generate a sentence from the words
        sentence = self.mycelium_generator._generate_mycelium_sentence(words)
        
        # Check the result
        self.assertIsInstance(sentence, self.mycelium_generator.MyceliumSentence)
        self.assertEqual(sentence.words, words)
        self.assertIsInstance(sentence.syntactic_structure, str)
        self.assertIsInstance(sentence.semantic_flow, dict)
        self.assertIsInstance(sentence.network_topology, str)
        self.assertIsInstance(sentence.temporal_pattern, str)
        self.assertIsInstance(sentence.consciousness_level, str)
        
        # Check that the sentence has meaningful content
        self.assertGreater(len(sentence.syntactic_structure), 0)
        self.assertGreater(len(sentence.network_topology), 0)
        self.assertGreater(len(sentence.temporal_pattern), 0)
        self.assertGreater(len(sentence.consciousness_level), 0)

class TestGardenConsciousnessIntegration(unittest.TestCase):
    """Test cases for the Garden Consciousness Integration"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.mycelium_language_generator import (
                MyceliumLanguageGenerator,
                GardenConsciousnessIntegration
            )
            self.mycelium_generator = MyceliumLanguageGenerator(network_size=500)
            self.garden_integration = GardenConsciousnessIntegration()
        except ImportError as e:
            self.skipTest(f"Garden Consciousness Integration not available: {e}")
    
    def test_garden_integration_initialization(self):
        """Test that the Garden Consciousness Integration initializes correctly"""
        self.assertIsNotNone(self.garden_integration)
        self.assertIsInstance(self.garden_integration.integration_protocols, dict)
        self.assertGreater(len(self.garden_integration.integration_protocols), 0)
        
        # Check that all consciousness forms have protocols
        expected_forms = ['plant_consciousness', 'fungal_consciousness', 'quantum_consciousness', 
                         'ecosystem_consciousness', 'shamanic_consciousness']
        for form in expected_forms:
            self.assertIn(form, self.garden_integration.integration_protocols)
    
    def test_integrate_consciousness_data(self):
        """Test integrating consciousness data"""
        # Create test consciousness data
        consciousness_data = {
            'consciousness_form': 'plant',
            'consciousness_level': 0.75,
            'coherence': 0.65,
            'connectivity': 0.8
        }
        
        # Integrate the data
        integrated_data = self.garden_integration.integrate_consciousness_data(consciousness_data)
        
        # Check the result
        self.assertIsInstance(integrated_data, dict)
        self.assertIn('integration_timestamp', integrated_data)
        self.assertIn('integration_protocol', integrated_data)
        self.assertIn('coherence_aligned', integrated_data)
        self.assertEqual(integrated_data['integration_protocol'], 'garden_of_consciousness_v2.0')
        
        # Check that coherence was aligned
        self.assertLessEqual(integrated_data['coherence_aligned'], consciousness_data['coherence'])

class TestFieldsFirstbornTranslator(unittest.TestCase):
    """Test cases for the Fields-Firstborn Translator"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.mycelium_language_generator import (
                FieldsFirstbornTranslator
            )
            self.fields_translator = FieldsFirstbornTranslator()
        except ImportError as e:
            self.skipTest(f"Fields-Firstborn Translator not available: {e}")
    
    def test_fields_translator_initialization(self):
        """Test that the Fields-Firstborn Translator initializes correctly"""
        self.assertIsNotNone(self.fields_translator)
        self.assertIsInstance(self.fields_translator.interform_translations, dict)
        self.assertGreater(len(self.fields_translator.interform_translations), 0)
        
        # Check that all universal interforms have translations
        expected_interforms = ['energy', 'electricity', 'water', 'rhythm', 'information', 'mycelium']
        for interform in expected_interforms:
            self.assertIn(interform, self.fields_translator.interform_translations)
    
    def test_translate_to_fields_firstborn(self):
        """Test translating consciousness data to Fields-Firstborn universal interforms"""
        # Create test consciousness data
        consciousness_data = {
            'consciousness_level': 0.85,
            'coherence': 0.75,
            'connectivity': 0.9
        }
        
        # Translate to Fields-Firstborn
        translated_data = self.fields_translator.translate_to_fields_firstborn(consciousness_data)
        
        # Check the result
        self.assertIsInstance(translated_data, dict)
        self.assertIn('translation_framework', translated_data)
        self.assertIn('universal_carriers', translated_data)
        self.assertIn('universal_integration', translated_data)
        self.assertEqual(translated_data['translation_framework'], 'fields_firstborn_universal_interforms')
        
        # Check that universal integration state is correct
        self.assertEqual(translated_data['universal_integration'], 'holistic_state')
        self.assertTrue(translated_data['awakened_garden_state'])

def main():
    """Run the tests"""
    unittest.main()

if __name__ == '__main__':
    main()