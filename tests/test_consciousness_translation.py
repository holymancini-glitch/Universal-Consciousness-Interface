#!/usr/bin/env python3
"""
Translation accuracy tests for the Consciousness Translation Matrix
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

class TestConsciousnessTranslationMatrix(unittest.TestCase):
    """Test cases for the Consciousness Translation Matrix"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.consciousness_translation_matrix import (
                ConsciousnessTranslationMatrix,
                ConsciousnessForm,
                TranslationMode,
                ConsciousnessRepresentation,
                TranslationResult
            )
            self.translation_matrix = ConsciousnessTranslationMatrix()
            self.ConsciousnessForm = ConsciousnessForm
            self.TranslationMode = TranslationMode
            self.ConsciousnessRepresentation = ConsciousnessRepresentation
            self.TranslationResult = TranslationResult
        except ImportError as e:
            self.skipTest(f"Consciousness Translation Matrix not available: {e}")
    
    def test_initialization(self):
        """Test that the Consciousness Translation Matrix initializes correctly"""
        self.assertIsNotNone(self.translation_matrix)
        self.assertIsInstance(self.translation_matrix.translation_engines, dict)
        self.assertIsInstance(self.translation_matrix.translation_history, list)
        self.assertIsNotNone(self.translation_matrix.matrix_optimizer)
        self.assertIsNotNone(self.translation_matrix.semantic_bridge)
        self.assertIsNotNone(self.translation_matrix.adaptive_translator)
        
        # Check that translation engines are initialized
        self.assertGreater(len(self.translation_matrix.translation_engines), 0)
        
        # Check that all consciousness forms have translation engines
        forms = list(self.ConsciousnessForm)
        for source_form in forms:
            for target_form in forms:
                if source_form != target_form:
                    pair = (source_form, target_form)
                    self.assertIn(pair, self.translation_matrix.translation_engines)
    
    def test_self_translation(self):
        """Test translating consciousness to the same form (self-translation)"""
        from datetime import datetime
        
        # Create a consciousness representation
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousnessForm.PLANT,
            data={'frequency': 12.3, 'amplitude': 0.67},
            consciousness_level=0.7,
            dimensional_state="STABLE",
            timestamp=datetime.now()
        )
        
        # Translate to the same form
        result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.PLANT
        )
        
        # Check the result
        self.assertIsInstance(result, self.TranslationResult)
        self.assertEqual(result.source_form, self.ConsciousnessForm.PLANT)
        self.assertEqual(result.target_form, self.ConsciousnessForm.PLANT)
        self.assertEqual(result.translated_data, source_data.data)
        self.assertEqual(result.translation_quality, 1.0)
        self.assertEqual(result.semantic_preservation, 1.0)
        self.assertEqual(result.consciousness_fidelity, 1.0)
        self.assertEqual(result.translation_mode, self.TranslationMode.ADAPTIVE)
    
    def test_plant_to_fungal_translation(self):
        """Test translating plant consciousness to fungal consciousness"""
        from datetime import datetime
        
        # Create a plant consciousness representation
        plant_data = {
            'frequency': 12.3,
            'amplitude': 0.67,
            'pattern': 'GROWTH_RHYTHM'
        }
        
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousnessForm.PLANT,
            data=plant_data,
            consciousness_level=0.7,
            dimensional_state="STABLE",
            timestamp=datetime.now()
        )
        
        # Translate to fungal consciousness
        result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.FUNGAL
        )
        
        # Check the result
        self.assertIsInstance(result, self.TranslationResult)
        self.assertEqual(result.source_form, self.ConsciousnessForm.PLANT)
        self.assertEqual(result.target_form, self.ConsciousnessForm.FUNGAL)
        self.assertIsInstance(result.translated_data, dict)
        self.assertGreaterEqual(result.translation_quality, 0.0)
        self.assertLessEqual(result.translation_quality, 1.0)
        self.assertGreaterEqual(result.semantic_preservation, 0.0)
        self.assertLessEqual(result.semantic_preservation, 1.0)
        self.assertGreaterEqual(result.consciousness_fidelity, 0.0)
        self.assertLessEqual(result.consciousness_fidelity, 1.0)
        
        # Check that translated data contains expected elements
        self.assertIn('translated_compounds', result.translated_data)
        self.assertIn('fungal_analogue', result.translated_data)
    
    def test_quantum_to_digital_translation(self):
        """Test translating quantum consciousness to digital consciousness"""
        from datetime import datetime
        
        # Create a quantum consciousness representation
        quantum_data = {
            'coherence': 0.85,
            'entanglement': 0.75,
            'superposition': True
        }
        
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousnessForm.QUANTUM,
            data=quantum_data,
            consciousness_level=0.8,
            dimensional_state="SUPERPOSITION",
            timestamp=datetime.now()
        )
        
        # Translate to digital consciousness
        result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.DIGITAL
        )
        
        # Check the result
        self.assertIsInstance(result, self.TranslationResult)
        self.assertEqual(result.source_form, self.ConsciousnessForm.QUANTUM)
        self.assertEqual(result.target_form, self.ConsciousnessForm.DIGITAL)
        self.assertIsInstance(result.translated_data, dict)
        self.assertGreaterEqual(result.translation_quality, 0.0)
        self.assertLessEqual(result.translation_quality, 1.0)
        
        # Check that translated data contains expected elements
        self.assertIn('digital_coherence', result.translated_data)
        self.assertIn('binary_entanglement', result.translated_data)
        self.assertIn('computational_superposition', result.translated_data)
    
    def test_translation_with_different_modes(self):
        """Test translation with different modes"""
        from datetime import datetime
        
        # Create a consciousness representation
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousForm.PLANT,
            data={'frequency': 12.3, 'amplitude': 0.67},
            consciousness_level=0.7,
            dimensional_state="STABLE",
            timestamp=datetime.now()
        )
        
        # Test direct translation mode
        direct_result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.FUNGAL,
            mode=self.TranslationMode.DIRECT
        )
        
        self.assertIsInstance(direct_result, self.TranslationResult)
        self.assertEqual(direct_result.translation_mode, self.TranslationMode.DIRECT)
        
        # Test adaptive translation mode
        adaptive_result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.FUNGAL,
            mode=self.TranslationMode.ADAPTIVE
        )
        
        self.assertIsInstance(adaptive_result, self.TranslationResult)
        self.assertEqual(adaptive_result.translation_mode, self.TranslationMode.ADAPTIVE)
        
        # Test symbiotic translation mode
        symbiotic_result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.FUNGAL,
            mode=self.TranslationMode.SYMBIOTIC
        )
        
        self.assertIsInstance(symbiotic_result, self.TranslationResult)
        self.assertEqual(symbiotic_result.translation_mode, self.TranslationMode.SYMBIOTIC)
    
    def test_translation_history(self):
        """Test that translations are added to history"""
        from datetime import datetime
        
        # Create a consciousness representation
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousnessForm.PLANT,
            data={'frequency': 12.3, 'amplitude': 0.67},
            consciousness_level=0.7,
            dimensional_state="STABLE",
            timestamp=datetime.now()
        )
        
        # Perform a translation
        result = self.translation_matrix.translate_consciousness(
            source_data=source_data,
            target_form=self.ConsciousnessForm.FUNGAL
        )
        
        # Check that translation was added to history
        self.assertGreater(len(self.translation_matrix.translation_history), 0)
        self.assertEqual(self.translation_matrix.translation_history[-1], result)

class TestTranslationEngine(unittest.TestCase):
    """Test cases for the Translation Engine"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.consciousness_translation_matrix import (
                TranslationEngine,
                ConsciousnessForm,
                ConsciousnessRepresentation
            )
            self.translation_engine = TranslationEngine(
                source_form=ConsciousnessForm.PLANT,
                target_form=ConsciousnessForm.FUNGAL
            )
            self.ConsciousnessForm = ConsciousnessForm
            self.ConsciousnessRepresentation = ConsciousnessRepresentation
        except ImportError as e:
            self.skipTest(f"Translation Engine not available: {e}")
    
    def test_translation_engine_initialization(self):
        """Test that the Translation Engine initializes correctly"""
        self.assertIsNotNone(self.translation_engine)
        self.assertEqual(self.translation_engine.source_form, self.ConsciousnessForm.PLANT)
        self.assertEqual(self.translation_engine.target_form, self.ConsciousnessForm.FUNGAL)
        self.assertIsNotNone(self.translation_engine.translation_algorithms)
        self.assertIsNotNone(self.translation_engine.mapping_functions)
    
    def test_direct_translation(self):
        """Test direct translation functionality"""
        from datetime import datetime
        
        # Create a consciousness representation
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousnessForm.PLANT,
            data={'frequency': 12.3, 'amplitude': 0.67, 'pattern': 'GROWTH_RHYTHM'},
            consciousness_level=0.7,
            dimensional_state="STABLE",
            timestamp=datetime.now()
        )
        
        # Perform direct translation
        result = self.translation_engine.direct_translation(source_data)
        
        # Check the result
        self.assertEqual(result.source_form, self.ConsciousnessForm.PLANT)
        self.assertEqual(result.target_form, self.ConsciousnessForm.FUNGAL)
        self.assertIsInstance(result.translated_data, dict)
        self.assertGreaterEqual(result.translation_quality, 0.0)
        self.assertLessEqual(result.translation_quality, 1.0)

class TestSemanticBridge(unittest.TestCase):
    """Test cases for the Semantic Bridge"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.consciousness_translation_matrix import (
                SemanticBridge
            )
            self.semantic_bridge = SemanticBridge()
        except ImportError as e:
            self.skipTest(f"Semantic Bridge not available: {e}")
    
    def test_semantic_bridge_initialization(self):
        """Test that the Semantic Bridge initializes correctly"""
        self.assertIsNotNone(self.semantic_bridge)
        self.assertIsNotNone(self.semantic_bridge.semantic_mappings)
        self.assertIsNotNone(self.semantic_bridge.contextual_adapters)
        
        # Check that semantic mappings are loaded
        self.assertGreater(len(self.semantic_bridge.semantic_mappings), 0)
    
    def test_map_semantic_concepts(self):
        """Test mapping semantic concepts between consciousness forms"""
        # Test mapping plant concepts to fungal concepts
        plant_concepts = {
            'GROWTH_RHYTHM': 0.8,
            'STRESS_RESPONSE': 0.6,
            'COMMUNICATION_PULSE': 0.7
        }
        
        mapped_concepts = self.semantic_bridge.map_semantic_concepts(
            plant_concepts, 
            'plant', 
            'fungal'
        )
        
        # Check the result
        self.assertIsInstance(mapped_concepts, dict)
        self.assertGreater(len(mapped_concepts), 0)
        
        # Check that all concepts have been mapped
        for concept, value in plant_concepts.items():
            self.assertIn(concept, mapped_concepts)

class TestAdaptiveTranslator(unittest.TestCase):
    """Test cases for the Adaptive Translator"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        try:
            from core.consciousness_translation_matrix import (
                AdaptiveTranslator,
                TranslationEngine,
                ConsciousnessForm,
                ConsciousnessRepresentation
            )
            self.adaptive_translator = AdaptiveTranslator()
            self.TranslationEngine = TranslationEngine
            self.ConsciousnessForm = ConsciousnessForm
            self.ConsciousnessRepresentation = ConsciousnessRepresentation
        except ImportError as e:
            self.skipTest(f"Adaptive Translator not available: {e}")
    
    def test_adaptive_translator_initialization(self):
        """Test that the Adaptive Translator initializes correctly"""
        self.assertIsNotNone(self.adaptive_translator)
        self.assertIsNotNone(self.adaptive_translator.adaptation_algorithms)
        self.assertIsNotNone(self.adaptive_translator.learning_models)
    
    def test_adaptive_translation(self):
        """Test adaptive translation functionality"""
        from datetime import datetime
        
        # Create a translation engine
        engine = self.TranslationEngine(
            source_form=self.ConsciousnessForm.PLANT,
            target_form=self.ConsciousnessForm.FUNGAL
        )
        
        # Create a consciousness representation
        source_data = self.ConsciousnessRepresentation(
            form=self.ConsciousnessForm.PLANT,
            data={'frequency': 12.3, 'amplitude': 0.67, 'pattern': 'GROWTH_RHYTHM'},
            consciousness_level=0.7,
            dimensional_state="STABLE",
            timestamp=datetime.now()
        )
        
        # Perform adaptive translation
        result = self.adaptive_translator.adaptive_translation(
            source_data, 
            self.ConsciousnessForm.FUNGAL, 
            engine
        )
        
        # Check the result
        self.assertEqual(result.source_form, self.ConsciousnessForm.PLANT)
        self.assertEqual(result.target_form, self.ConsciousnessForm.FUNGAL)
        self.assertIsInstance(result.translated_data, dict)
        self.assertGreaterEqual(result.translation_quality, 0.0)
        self.assertLessEqual(result.translation_quality, 1.0)

def main():
    """Run the tests"""
    unittest.main()

if __name__ == '__main__':
    main()