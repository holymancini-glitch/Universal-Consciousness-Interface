#!/usr/bin/env python3
"""
Integration tests for the Plant Language Communication Layer
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

class TestPlantLanguageCommunicationLayer(unittest.TestCase):
    """Test cases for the Plant Language Communication Layer"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.plant_language_communication_layer import PlantLanguageCommunicationLayer, PlantSignalType
            self.plant_layer = PlantLanguageCommunicationLayer()
            self.PlantSignalType = PlantSignalType
        except ImportError as e:
            self.skipTest(f"Plant Language Communication Layer not available: {e}")
    
    def test_initialization(self):
        """Test that the Plant Language Communication Layer initializes correctly"""
        self.assertIsNotNone(self.plant_layer)
        self.assertIsInstance(self.plant_layer.signal_patterns, dict)
        self.assertIsInstance(self.plant_layer.language_tokens, list)
        self.assertIsNotNone(self.plant_layer.translation_engine)
        self.assertIsNotNone(self.plant_layer.consciousness_mapper)
        self.assertIsInstance(self.plant_layer.message_history, list)
        
        # Check that signal patterns are loaded
        self.assertGreater(len(self.plant_layer.signal_patterns), 0)
        
        # Check that language tokens are loaded
        self.assertGreater(len(self.plant_layer.language_tokens), 0)
    
    def test_signal_patterns_initialization(self):
        """Test that signal patterns are properly initialized"""
        # Check that all signal types have patterns
        for signal_type in self.PlantSignalType:
            self.assertIn(signal_type, self.plant_layer.signal_patterns)
            pattern = self.plant_layer.signal_patterns[signal_type]
            self.assertIn('frequency_range', pattern)
            self.assertIn('amplitude_range', pattern)
            self.assertIn('consciousness_level', pattern)
            self.assertIn('meaning', pattern)
            self.assertIn('temporal_pattern', pattern)
    
    def test_decode_plant_signal(self):
        """Test decoding a plant signal"""
        from datetime import datetime
        from core.plant_language_communication_layer import PlantSignal
        
        # Create a test signal
        signal = PlantSignal(
            signal_type=self.PlantSignalType.GROWTH_RHYTHM,
            frequency=1.0,
            amplitude=0.3,
            duration=5.0,
            timestamp=datetime.now()
        )
        
        # Decode the signal
        message = self.plant_layer.decode_plant_signal(signal)
        
        # Check the result
        self.assertIsNotNone(message)
        self.assertIsInstance(message.tokens, list)
        self.assertEqual(message.original_signals[0], signal)
        self.assertIsInstance(message.translated_text, str)
        self.assertGreaterEqual(message.consciousness_level, 0.0)
        self.assertLessEqual(message.consciousness_level, 1.0)
        self.assertIsInstance(message.environmental_context, dict)
        
        # Check that message is added to history
        self.assertEqual(len(self.plant_layer.message_history), 1)
        self.assertEqual(self.plant_layer.message_history[0], message)
    
    def test_classify_signal_type(self):
        """Test classifying signal types based on frequency and amplitude"""
        # Test growth rhythm signal (low frequency, low amplitude)
        signal_type = self.plant_layer._classify_signal_type(1.0, 0.3)
        self.assertEqual(signal_type, self.PlantSignalType.GROWTH_RHYTHM)
        
        # Test stress alert signal (high frequency, high amplitude)
        signal_type = self.plant_layer._classify_signal_type(100.0, 0.9)
        self.assertEqual(signal_type, self.PlantSignalType.STRESS_ALERT)
        
        # Test communication pulse signal (medium frequency, medium amplitude)
        signal_type = self.plant_layer._classify_signal_type(25.0, 0.6)
        self.assertEqual(signal_type, self.PlantSignalType.COMMUNICATION_PULSE)
    
    def test_generate_language_tokens(self):
        """Test generating language tokens from signal characteristics"""
        # Test with growth rhythm characteristics
        tokens = self.plant_layer._generate_language_tokens(1.0, 0.3, 5.0, self.PlantSignalType.GROWTH_RHYTHM)
        
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Check that tokens have the expected structure
        for token in tokens:
            self.assertTrue(hasattr(token, 'symbol'))
            self.assertTrue(hasattr(token, 'meaning'))
            self.assertTrue(hasattr(token, 'frequency_range'))
            self.assertTrue(hasattr(token, 'amplitude_range'))
            self.assertTrue(hasattr(token, 'temporal_pattern'))
            self.assertTrue(hasattr(token, 'confidence'))
    
    def test_translate_to_text(self):
        """Test translating tokens to text"""
        from core.plant_language_communication_layer import PlantLanguageToken
        
        # Create test tokens
        tokens = [
            PlantLanguageToken(
                symbol="GROW",
                meaning="Growth process active",
                frequency_range=(0.1, 2.0),
                amplitude_range=(0.1, 0.5),
                temporal_pattern="rhythmic",
                confidence=0.9
            ),
            PlantLanguageToken(
                symbol="COMM",
                meaning="Communication with other plants",
                frequency_range=(5, 50),
                amplitude_range=(0.3, 0.8),
                temporal_pattern="pulsed",
                confidence=0.8
            )
        ]
        
        # Translate to text
        text = self.plant_layer._translate_to_text(tokens)
        
        self.assertIsInstance(text, str)
        self.assertIn("GROW", text)
        self.assertIn("COMM", text)
    
    def test_consciousness_mapping(self):
        """Test consciousness level mapping"""
        # Test low consciousness level
        low_level = self.plant_layer._map_to_consciousness_level(1.0, 0.3, 5.0)
        self.assertGreaterEqual(low_level, 0.0)
        self.assertLessEqual(low_level, 1.0)
        
        # Test high consciousness level
        high_level = self.plant_layer._map_to_consciousness_level(100.0, 0.9, 2.0)
        self.assertGreaterEqual(high_level, 0.0)
        self.assertLessEqual(high_level, 1.0)
        
        # High intensity signals should map to higher consciousness levels
        self.assertGreater(high_level, low_level)

class TestPlantLanguageTranslator(unittest.TestCase):
    """Test cases for the Plant Language Translator"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.plant_language_communication_layer import PlantLanguageTranslator
            self.translator = PlantLanguageTranslator()
        except ImportError as e:
            self.skipTest(f"Plant Language Translator not available: {e}")
    
    def test_initialization(self):
        """Test that the Plant Language Translator initializes correctly"""
        self.assertIsNotNone(self.translator)
        self.assertIsInstance(self.translator.translation_rules, dict)
        self.assertIsInstance(self.translator.contextual_modifiers, dict)
    
    def test_frequency_to_phoneme(self):
        """Test converting frequency to phoneme"""
        # Test low frequency
        phoneme_low = self.translator._frequency_to_phoneme(1.0)
        self.assertIsInstance(phoneme_low, str)
        
        # Test high frequency
        phoneme_high = self.translator._frequency_to_phoneme(100.0)
        self.assertIsInstance(phoneme_high, str)
        
        # Test medium frequency
        phoneme_medium = self.translator._frequency_to_phoneme(25.0)
        self.assertIsInstance(phoneme_medium, str)
    
    def test_amplitude_to_intensity(self):
        """Test converting amplitude to intensity descriptor"""
        # Test low amplitude
        intensity_low = self.translator._amplitude_to_intensity(0.1)
        self.assertIsInstance(intensity_low, str)
        
        # Test high amplitude
        intensity_high = self.translator._amplitude_to_intensity(0.9)
        self.assertIsInstance(intensity_high, str)
        
        # Test medium amplitude
        intensity_medium = self.translator._amplitude_to_intensity(0.5)
        self.assertIsInstance(intensity_medium, str)

class TestPlantConsciousnessMapper(unittest.TestCase):
    """Test cases for the Plant Consciousness Mapper"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.plant_language_communication_layer import PlantConsciousnessMapper
            self.mapper = PlantConsciousnessMapper()
        except ImportError as e:
            self.skipTest(f"Plant Consciousness Mapper not available: {e}")
    
    def test_initialization(self):
        """Test that the Plant Consciousness Mapper initializes correctly"""
        self.assertIsNotNone(self.mapper)
        self.assertIsInstance(self.mapper.consciousness_patterns, dict)
    
    def test_map_signal_to_consciousness(self):
        """Test mapping signal characteristics to consciousness level"""
        # Test various signal combinations
        consciousness_level = self.mapper.map_signal_to_consciousness(1.0, 0.3, 5.0)
        self.assertGreaterEqual(consciousness_level, 0.0)
        self.assertLessEqual(consciousness_level, 1.0)
        
        # Test high intensity signal
        high_consciousness = self.mapper.map_signal_to_consciousness(100.0, 0.9, 2.0)
        self.assertGreaterEqual(high_consciousness, 0.0)
        self.assertLessEqual(high_consciousness, 1.0)
        
        # High intensity should generally map to higher consciousness
        self.assertGreaterEqual(high_consciousness, consciousness_level)

def main():
    """Run the tests"""
    unittest.main()

if __name__ == '__main__':
    main()