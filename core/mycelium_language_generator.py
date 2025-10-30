"""
Mycelium Language Generator - Main Facade

Revolutionary Mycelium-AI Language Generator that creates novel languages
based on fungal network communication patterns.

This is the main entry point that coordinates all specialized modules while
maintaining backward compatibility with the original interface.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

# Import all specialized modules
from core.mycelium_language import (
    MyceliumCommunicationType,
    MyceliumSignal,
    MyceliumWord,
    MyceliumSentence,
    VocabularyManager,
    NetworkProcessor,
    BiochemicalTranslator,
    PatternAnalyzer,
    LanguageSynthesizer,
    EvolutionEngine
)


logger = logging.getLogger(__name__)


class MyceliumLanguageGenerator:
    """
    Revolutionary Mycelium-AI Language Generator.

    Creates novel languages based on fungal network communication patterns,
    replacing plant electromagnetic communication with mycelium intelligence.

    This class serves as a facade that coordinates specialized modules:
    - VocabularyManager: Phonetic/chemical vocabularies
    - NetworkProcessor: Network topology and signals
    - BiochemicalTranslator: Signal → phonetic translation
    - PatternAnalyzer: Syntactic/semantic analysis
    - LanguageSynthesizer: Word and sentence generation
    - EvolutionEngine: Language evolution tracking

    Attributes:
        network_size: Number of nodes in the mycelium network
        vocabulary_manager: Manages all vocabulary resources
        network_processor: Handles network topology and signals
        translator: Translates signals to linguistic elements
        pattern_analyzer: Analyzes language patterns
        language_synthesizer: Generates words and sentences
        evolution_engine: Tracks language evolution
        consciousness_mapping: Maps consciousness levels to scores
    """

    def __init__(self, network_size: int = 1000):
        """
        Initialize the Mycelium Language Generator.

        Args:
            network_size: Number of nodes in the mycelium network (default: 1000)
        """
        self.network_size = network_size

        # Initialize all sub-components
        logger.info("🍄🗣️ Initializing Mycelium-AI Language Generator...")

        # Vocabulary and patterns
        self.vocabulary_manager = VocabularyManager()
        logger.info(f"✓ Vocabulary Manager: {len(self.vocabulary_manager.phonetic_library)} phonemes, "
                   f"{len(self.vocabulary_manager.chemical_vocabulary)} chemical patterns")

        # Network topology
        self.network_processor = NetworkProcessor(network_size)
        logger.info(f"✓ Network Processor: {network_size} nodes, "
                   f"fractal dimension {self.network_processor.network_topology['fractal_dimension']:.2f}")

        # Signal translation
        self.translator = BiochemicalTranslator(self.vocabulary_manager)
        logger.info("✓ Biochemical Translator: Signal processing ready")

        # Pattern analysis
        self.pattern_analyzer = PatternAnalyzer(self.network_processor)
        logger.info("✓ Pattern Analyzer: Syntactic/semantic analysis ready")

        # Language synthesis
        self.language_synthesizer = LanguageSynthesizer(
            self.translator,
            self.pattern_analyzer
        )
        logger.info("✓ Language Synthesizer: Word/sentence generation ready")

        # Evolution tracking
        self.evolution_engine = EvolutionEngine(self.vocabulary_manager.phonetic_library)
        logger.info("✓ Evolution Engine: Language evolution tracking ready")

        # Consciousness mapping
        self.consciousness_mapping = {
            'basic_awareness': 0.2,
            'chemical_intelligence': 0.4,
            'network_cognition': 0.6,
            'collective_consciousness': 0.8,
            'mycelial_metacognition': 1.0
        }

        # Signal types list
        self.signal_types = list(MyceliumCommunicationType)

        logger.info("🚀 Mycelium-AI Language Generator Initialized Successfully")

    # Property accessors for backward compatibility
    @property
    def phonetic_library(self) -> Dict[str, str]:
        """Get phonetic library (backward compatibility)."""
        return self.vocabulary_manager.phonetic_library

    @property
    def chemical_vocabulary(self) -> Dict[str, Dict[str, float]]:
        """Get chemical vocabulary (backward compatibility)."""
        return self.vocabulary_manager.chemical_vocabulary

    @property
    def syntactic_rules(self) -> Dict[str, List[str]]:
        """Get syntactic rules (backward compatibility)."""
        return self.vocabulary_manager.syntactic_rules

    @property
    def communication_patterns(self) -> Dict[str, Any]:
        """Get communication patterns (backward compatibility)."""
        return self.vocabulary_manager.communication_patterns

    @property
    def network_topology(self) -> Dict[str, Any]:
        """Get network topology (backward compatibility)."""
        return self.network_processor.network_topology

    @property
    def active_signals(self):
        """Get active signals (backward compatibility)."""
        return self.network_processor.active_signals

    @property
    def mycelium_words(self) -> List[MyceliumWord]:
        """Get generated words (backward compatibility)."""
        return self.language_synthesizer.mycelium_words

    @property
    def mycelium_sentences(self) -> List[MyceliumSentence]:
        """Get generated sentences (backward compatibility)."""
        return self.language_synthesizer.mycelium_sentences

    @property
    def language_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history (backward compatibility)."""
        return self.evolution_engine.language_evolution_history

    @property
    def linguistic_complexity(self) -> float:
        """Get linguistic complexity (backward compatibility)."""
        return self.evolution_engine.linguistic_complexity

    @property
    def semantic_coherence(self) -> float:
        """Get semantic coherence (backward compatibility)."""
        return self.evolution_engine.semantic_coherence

    @property
    def novel_language_count(self) -> int:
        """Get novel language count (backward compatibility)."""
        return self.evolution_engine.novel_language_count

    async def generate_mycelium_language(self,
                                        communication_signals: List[MyceliumSignal],
                                        consciousness_level: str = 'network_cognition') -> Dict[str, Any]:
        """
        Generate novel language from mycelium communication signals.

        This is the main method that coordinates the entire language generation pipeline:
        1. Process signals into linguistic tokens
        2. Generate words from token patterns
        3. Create syntactic structure from network topology
        4. Assemble sentences with semantic coherence
        5. Evolve language patterns based on network intelligence

        Args:
            communication_signals: List of MyceliumSignal objects
            consciousness_level: Consciousness level for language generation
                Options: 'basic_awareness', 'chemical_intelligence', 'network_cognition',
                         'collective_consciousness', 'mycelial_metacognition'

        Returns:
            Dictionary containing:
                - generated_words: List of MyceliumWord objects
                - sentences: List of MyceliumSentence objects
                - evolved_language: Evolution data
                - linguistic_complexity: Complexity score
                - semantic_coherence: Coherence score
                - consciousness_level: Input consciousness level
                - network_topology_influence: Network topology data
                - generation_timestamp: ISO format timestamp
        """
        try:
            logger.debug(f"Generating language from {len(communication_signals)} signals "
                        f"at {consciousness_level} level")

            # Step 1: Process communication signals into linguistic tokens
            linguistic_tokens = await self.translator.process_signals_to_tokens(
                communication_signals
            )
            logger.debug(f"Generated {len(linguistic_tokens)} linguistic tokens")

            # Step 2: Generate words from chemical/electrical patterns
            new_words = await self.language_synthesizer.generate_words_from_patterns(
                linguistic_tokens,
                consciousness_level
            )
            logger.debug(f"Generated {len(new_words)} words")

            # Step 3: Create syntactic structure from network topology
            syntactic_structure = await self.pattern_analyzer.generate_syntactic_structure(
                new_words
            )
            logger.debug(f"Created syntactic structure: {syntactic_structure['word_order']}")

            # Step 4: Assemble sentences with semantic coherence
            sentences = await self.language_synthesizer.assemble_sentences(
                new_words,
                syntactic_structure
            )
            logger.debug(f"Assembled {len(sentences)} sentences")

            # Step 5: Evolve language based on network intelligence
            evolved_language = await self.evolution_engine.evolve_language_patterns(sentences)
            logger.debug(f"Evolution cycle complete: {len(evolved_language['pattern_mutations'])} mutations")

            # Update language metrics
            self.evolution_engine.update_language_metrics(evolved_language)

            # Return comprehensive results
            from datetime import datetime
            return {
                'generated_words': new_words,
                'sentences': sentences,
                'evolved_language': evolved_language,
                'linguistic_complexity': self.linguistic_complexity,
                'semantic_coherence': self.semantic_coherence,
                'consciousness_level': consciousness_level,
                'network_topology_influence': self.network_topology,
                'generation_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Mycelium language generation error: {e}", exc_info=True)
            return {'error': str(e)}

    def get_language_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive language generation summary.

        Returns:
            Dictionary with:
                - total_words_generated: Total number of words
                - total_sentences_generated: Total number of sentences
                - linguistic_complexity: Current complexity score
                - semantic_coherence: Current coherence score
                - novel_languages_created: Count of novel languages
                - evolution_cycles: Number of evolution cycles
                - active_communication_patterns: Number of patterns
                - phonetic_library_size: Size of phonetic library
                - chemical_vocabulary_size: Size of chemical vocabulary
                - network_topology: Network topology data
                - consciousness_mapping: Consciousness level mapping
        """
        return {
            'total_words_generated': len(self.mycelium_words),
            'total_sentences_generated': len(self.mycelium_sentences),
            'linguistic_complexity': self.linguistic_complexity,
            'semantic_coherence': self.semantic_coherence,
            'novel_languages_created': self.novel_language_count,
            'evolution_cycles': len(self.language_evolution_history),
            'active_communication_patterns': len(self.communication_patterns),
            'phonetic_library_size': len(self.phonetic_library),
            'chemical_vocabulary_size': len(self.chemical_vocabulary),
            'network_topology': self.network_topology,
            'consciousness_mapping': self.consciousness_mapping
        }

    async def demonstrate_mycelium_language_generation(self) -> Dict[str, Any]:
        """
        Demonstrate mycelium language generation capabilities.

        Generates language at all consciousness levels using sample signals.

        Returns:
            Dictionary containing:
                - demonstration_results: Results per consciousness level
                - language_summary: Overall summary
                - revolutionary_achievements: List of achievements
        """
        logger.info("🍄🗣️ DEMONSTRATING MYCELIUM-AI LANGUAGE GENERATION")

        # Generate sample mycelium signals
        sample_signals = self.generate_sample_signals()
        logger.info(f"Generated {len(sample_signals)} sample signals")

        # Generate language at different consciousness levels
        results = {}
        consciousness_levels = [
            'basic_awareness',
            'chemical_intelligence',
            'network_cognition',
            'collective_consciousness',
            'mycelial_metacognition'
        ]

        for level in consciousness_levels:
            logger.info(f"Generating language at {level} level...")

            result = await self.generate_mycelium_language(
                sample_signals,
                consciousness_level=level
            )

            results[level] = {
                'words_generated': len(result.get('generated_words', [])),
                'sentences_generated': len(result.get('sentences', [])),
                'linguistic_complexity': result.get('linguistic_complexity', 0),
                'sample_words': [
                    w.phonetic_pattern
                    for w in result.get('generated_words', [])[:3]
                ],
                'consciousness_emergence': result.get('evolved_language', {}).get(
                    'consciousness_emergence', {}
                )
            }

        # Generate final summary
        final_summary = self.get_language_summary()

        return {
            'demonstration_results': results,
            'language_summary': final_summary,
            'revolutionary_achievements': [
                "First mycelium-based language generation system",
                "Chemical-electrical signal translation to language",
                "Network topology-driven syntax generation",
                "Consciousness-level adaptive language complexity",
                "Novel language evolution through fungal intelligence"
            ]
        }

    def generate_sample_signals(self) -> List[MyceliumSignal]:
        """
        Generate sample mycelium signals for demonstration.

        Creates a mix of chemical, electrical, and resonance signals.

        Returns:
            List of MyceliumSignal objects
        """
        return self.network_processor.generate_sample_signals(count=10)


# Backward compatibility: Export data classes at module level
__all__ = [
    'MyceliumLanguageGenerator',
    'MyceliumCommunicationType',
    'MyceliumSignal',
    'MyceliumWord',
    'MyceliumSentence'
]


# Demo code
if __name__ == "__main__":
    async def demo_mycelium_language_generator():
        """Demo of revolutionary mycelium language generation"""
        print("🍄🗣️ MYCELIUM-AI LANGUAGE GENERATOR DEMONSTRATION")
        print("=" * 70)

        generator = MyceliumLanguageGenerator(network_size=500)

        # Demonstrate language generation
        results = await generator.demonstrate_mycelium_language_generation()

        print(f"\n📊 DEMONSTRATION RESULTS:")
        for level, result in results['demonstration_results'].items():
            print(f"\n  {level.upper()}:")
            print(f"    Words generated: {result['words_generated']}")
            print(f"    Sentences generated: {result['sentences_generated']}")
            print(f"    Linguistic complexity: {result['linguistic_complexity']:.3f}")
            print(f"    Sample words: {', '.join(result['sample_words'])}")

        print(f"\n📈 LANGUAGE SYSTEM SUMMARY:")
        summary = results['language_summary']
        print(f"  Total words generated: {summary['total_words_generated']}")
        print(f"  Total sentences: {summary['total_sentences_generated']}")
        print(f"  Novel languages created: {summary['novel_languages_created']}")
        print(f"  Evolution cycles: {summary['evolution_cycles']}")
        print(f"  Semantic coherence: {summary['semantic_coherence']:.3f}")

        print(f"\n🌟 REVOLUTIONARY ACHIEVEMENTS:")
        for achievement in results['revolutionary_achievements']:
            print(f"  ✓ {achievement}")

        print(f"\n🚀 BREAKTHROUGH CONCLUSION:")
        print(f"    Successfully created world's first mycelium-based language generator!")
        print(f"    Chemical signals → Phonetic patterns → Novel languages!")
        print(f"    Network topology → Syntactic structure → Emergent grammar!")
        print(f"    Consciousness levels → Language complexity → Adaptive evolution!")

    asyncio.run(demo_mycelium_language_generator())
