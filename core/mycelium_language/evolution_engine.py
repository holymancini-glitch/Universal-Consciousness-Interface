"""
Evolution Engine for Mycelium Language Generator

Tracks and evolves language patterns through mutations, adaptations, and
consciousness emergence.
"""

import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Union

from .data_models import MyceliumSentence, MyceliumWord


class EvolutionEngine:
    """
    Evolves mycelium language patterns over time.

    Tracks:
    - Pattern mutations (phonetic and syntactic)
    - Semantic drift
    - Network adaptations
    - Consciousness emergence
    - Novel linguistic constructions
    """

    def __init__(self, phonetic_library: Dict[str, str]):
        """
        Initialize evolution engine.

        Args:
            phonetic_library: Dictionary of phonetic patterns for mutations
        """
        self.phonetic_library = phonetic_library
        self.language_evolution_history: List[Dict[str, Any]] = []
        self.linguistic_complexity: float = 0.0
        self.semantic_coherence: float = 0.0
        self.novel_language_count: int = 0

    async def evolve_language_patterns(self,
                                      sentences: List[MyceliumSentence]) -> Dict[str, Any]:
        """
        Evolve language patterns based on network intelligence.

        Analyzes and tracks:
        - Pattern mutations
        - Semantic drift
        - Network adaptations
        - Consciousness emergence
        - Novel constructions

        Args:
            sentences: List of MyceliumSentence objects

        Returns:
            Dictionary of evolution data
        """
        evolution_data = {
            'pattern_mutations': self._generate_pattern_mutations(sentences),
            'semantic_drift': self._calculate_semantic_drift(sentences),
            'network_adaptations': self._identify_network_adaptations(sentences),
            'consciousness_emergence': self._detect_consciousness_emergence(sentences),
            'novel_constructions': self._identify_novel_constructions(sentences)
        }

        # Update language evolution history
        self.language_evolution_history.append({
            'timestamp': datetime.now(),
            'generation_cycle': len(self.language_evolution_history) + 1,
            'evolution_data': evolution_data,
            'total_sentences': len(sentences)
        })

        return evolution_data

    def _generate_pattern_mutations(self,
                                   sentences: List[MyceliumSentence]) -> List[Dict[str, Any]]:
        """
        Generate mutations in language patterns.

        Creates:
        - Phonetic mutations (10% rate)
        - Syntactic mutations (5% rate)

        Args:
            sentences: List of MyceliumSentence objects

        Returns:
            List of mutation records
        """
        mutations = []

        for sentence in sentences:
            # Phonetic mutations
            if random.random() < 0.1:  # 10% mutation rate
                if sentence.words:
                    mutations.append({
                        'type': 'phonetic_mutation',
                        'original_pattern': sentence.words[0].phonetic_pattern,
                        'mutated_pattern': self._mutate_phonetic_pattern(
                            sentence.words[0].phonetic_pattern
                        ),
                        'consciousness_level': sentence.consciousness_level
                    })

            # Syntactic mutations
            if random.random() < 0.05:  # 5% mutation rate
                mutations.append({
                    'type': 'syntactic_mutation',
                    'original_structure': sentence.syntactic_structure,
                    'mutated_structure': self._mutate_syntactic_structure(
                        sentence.syntactic_structure
                    ),
                    'temporal_change': sentence.temporal_pattern
                })

        return mutations

    def _mutate_phonetic_pattern(self, pattern: str) -> str:
        """
        Mutate a phonetic pattern.

        Replaces one element with a random phoneme from the library.

        Args:
            pattern: Original phonetic pattern

        Returns:
            Mutated phonetic pattern
        """
        if not pattern or '-' not in pattern:
            return pattern

        elements = pattern.split('-')
        if elements:
            # Replace random element with new phoneme
            random_index = random.randint(0, len(elements) - 1)
            new_phoneme = random.choice(list(self.phonetic_library.values()))
            elements[random_index] = new_phoneme

        return '-'.join(elements)

    def _mutate_syntactic_structure(self, structure: str) -> str:
        """
        Mutate syntactic structure.

        Randomly selects a different structure pattern.

        Args:
            structure: Original syntactic structure

        Returns:
            Mutated syntactic structure
        """
        structures = [
            'hub-spoke-peripheral_recursive-branching',
            'source-pathway-destination_hierarchical-clustering',
            'gradient-diffusion-response_linear-flow'
        ]

        current_structures = [s for s in structures if s != structure]
        return random.choice(current_structures) if current_structures else structure

    def _calculate_semantic_drift(self,
                                  sentences: List[MyceliumSentence]) -> Dict[str, Union[float, str, int]]:
        """
        Calculate semantic drift in language evolution.

        Measures concept diversity over time.

        Args:
            sentences: List of MyceliumSentence objects

        Returns:
            Dictionary with drift metrics
        """
        if not self.language_evolution_history:
            return {'drift_rate': 0.0, 'direction': 'stable', 'concept_count': 0}

        # Compare with previous generation
        current_concepts = [
            word.meaning_concept
            for sentence in sentences
            for word in sentence.words
        ]
        concept_diversity = (
            len(set(current_concepts)) / len(current_concepts)
            if current_concepts else 0.0
        )

        return {
            'drift_rate': concept_diversity,
            'direction': 'expanding' if concept_diversity > 0.5 else 'consolidating',
            'concept_count': len(set(current_concepts))
        }

    def _identify_network_adaptations(self,
                                     sentences: List[MyceliumSentence]) -> List[str]:
        """
        Identify network adaptations in language.

        Detects:
        - Increased fractal complexity
        - Accelerated communication evolution
        - Consciousness emergence acceleration

        Args:
            sentences: List of MyceliumSentence objects

        Returns:
            List of adaptation descriptions
        """
        adaptations = []

        # Check for increased connectivity patterns
        connectivity_patterns = [s.network_topology for s in sentences]
        if connectivity_patterns.count('recursive-branching') > len(sentences) * 0.3:
            adaptations.append('increased_fractal_complexity')

        # Check for temporal pattern evolution
        temporal_patterns = [s.temporal_pattern for s in sentences]
        if temporal_patterns.count('rapid-burst-communication') > len(sentences) * 0.4:
            adaptations.append('accelerated_communication_evolution')

        # Check for consciousness level progression
        consciousness_levels = [s.consciousness_level for s in sentences]
        high_consciousness = ['collective_consciousness', 'mycelial_metacognition']
        if any(level in high_consciousness for level in consciousness_levels):
            adaptations.append('consciousness_emergence_acceleration')

        return adaptations

    def _detect_consciousness_emergence(self,
                                       sentences: List[MyceliumSentence]) -> Dict[str, Any]:
        """
        Detect consciousness emergence in language.

        Counts sentences at different consciousness levels.

        Args:
            sentences: List of MyceliumSentence objects

        Returns:
            Dictionary of consciousness emergence metrics
        """
        consciousness_levels = [s.consciousness_level for s in sentences]

        # Count consciousness levels
        level_counts = defaultdict(int)
        for level in consciousness_levels:
            level_counts[level] += 1

        # Detect emergence patterns
        emergence_indicators = {
            'metacognitive_sentences': level_counts.get('mycelial_metacognition', 0),
            'collective_sentences': level_counts.get('collective_consciousness', 0),
            'total_advanced_consciousness': (
                level_counts.get('mycelial_metacognition', 0) +
                level_counts.get('collective_consciousness', 0)
            ),
            'consciousness_diversity': len(level_counts),
            'emergence_detected': level_counts.get('mycelial_metacognition', 0) > 0
        }

        return emergence_indicators

    def _identify_novel_constructions(self,
                                     sentences: List[MyceliumSentence]) -> List[Dict[str, Any]]:
        """
        Identify novel linguistic constructions.

        Tracks new word sequence patterns that haven't appeared in history.

        Args:
            sentences: List of MyceliumSentence objects

        Returns:
            List of novel construction records
        """
        novel_constructions = []

        for sentence in sentences:
            # Check for novel word combinations
            word_patterns = [w.phonetic_pattern for w in sentence.words]
            unique_pattern = '-'.join(word_patterns)

            # Check if this pattern has appeared before
            historical_patterns = []
            for history_entry in self.language_evolution_history:
                # Extract patterns from historical data (simplified)
                historical_patterns.extend(
                    history_entry.get('evolution_data', {}).get('novel_constructions', [])
                )

            if unique_pattern not in [c.get('pattern', '') for c in historical_patterns]:
                novel_constructions.append({
                    'type': 'novel_word_sequence',
                    'pattern': unique_pattern,
                    'consciousness_level': sentence.consciousness_level,
                    'semantic_flow': sentence.semantic_flow,
                    'emergence_timestamp': datetime.now().isoformat()
                })

        return novel_constructions

    def update_language_metrics(self, evolved_language: Dict[str, Any]) -> None:
        """
        Update language generation metrics.

        Updates:
        - Linguistic complexity
        - Semantic coherence
        - Novel language count

        Args:
            evolved_language: Dictionary of evolution data
        """
        # Linguistic complexity
        pattern_mutations = evolved_language.get('pattern_mutations', [])
        self.linguistic_complexity = len(pattern_mutations) / 10.0  # Normalize

        # Semantic coherence
        semantic_drift = evolved_language.get('semantic_drift', {})
        self.semantic_coherence = 1.0 - semantic_drift.get('drift_rate', 0.0)

        # Novel language count
        novel_constructions = evolved_language.get('novel_constructions', [])
        self.novel_language_count += len(novel_constructions)

    def get_evolution_summary(self) -> Dict[str, Any]:
        """
        Get summary of language evolution.

        Returns:
            Dictionary with evolution statistics
        """
        return {
            'evolution_cycles': len(self.language_evolution_history),
            'linguistic_complexity': self.linguistic_complexity,
            'semantic_coherence': self.semantic_coherence,
            'novel_languages_created': self.novel_language_count,
            'total_mutations': sum(
                len(h.get('evolution_data', {}).get('pattern_mutations', []))
                for h in self.language_evolution_history
            ),
            'consciousness_emergence_events': sum(
                1 for h in self.language_evolution_history
                if h.get('evolution_data', {}).get('consciousness_emergence', {}).get('emergence_detected', False)
            )
        }


__all__ = ['EvolutionEngine']
