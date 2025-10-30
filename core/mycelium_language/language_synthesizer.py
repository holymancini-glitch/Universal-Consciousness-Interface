"""
Language Synthesizer for Mycelium Language Generator

Generates words and assembles sentences from linguistic tokens, combining
biochemical translation with pattern analysis.
"""

from collections import defaultdict
from typing import Dict, List, Any
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .data_models import MyceliumWord, MyceliumSentence
from .biochemical_translator import BiochemicalTranslator
from .pattern_analyzer import PatternAnalyzer


class LanguageSynthesizer:
    """
    Synthesizes mycelium words and sentences from linguistic tokens.

    Combines:
    - Biochemical translation (signals → tokens → phonetics)
    - Pattern analysis (syntactic structure, semantic relations)
    - Word generation (tokens → words with meaning)
    - Sentence assembly (words → structured sentences)
    """

    def __init__(self,
                 translator: BiochemicalTranslator,
                 pattern_analyzer: PatternAnalyzer):
        """
        Initialize language synthesizer.

        Args:
            translator: BiochemicalTranslator for signal processing
            pattern_analyzer: PatternAnalyzer for structure generation
        """
        self.translator = translator
        self.pattern_analyzer = pattern_analyzer
        self.mycelium_words: List[MyceliumWord] = []
        self.mycelium_sentences: List[MyceliumSentence] = []

    async def generate_words_from_patterns(self,
                                          tokens: List[Dict[str, Any]],
                                          consciousness_level: str) -> List[MyceliumWord]:
        """
        Generate mycelium words from linguistic tokens.

        Groups tokens semantically and creates words with:
        - Phonetic patterns
        - Chemical signatures
        - Electrical signatures
        - Meaning concepts

        Args:
            tokens: List of linguistic token dictionaries
            consciousness_level: Consciousness level for meaning derivation

        Returns:
            List of MyceliumWord objects
        """
        words = []

        # Group tokens by semantic similarity
        semantic_groups = self._group_tokens_semantically(tokens)

        for group_name, token_group in semantic_groups.items():
            # Create word from token group
            phonetic_pattern = self.translator.combine_phonetic_patterns(
                [t['phonetic_root'] for t in token_group]
            )

            # Generate chemical signature
            chemical_signature = self.translator.generate_chemical_signature(token_group)

            # Calculate electrical signature
            electrical_signature = self.translator.calculate_electrical_signature(token_group)

            # Determine meaning concept
            meaning_concept = self.translator.derive_meaning_concept(
                token_group,
                consciousness_level
            )

            # Create mycelium word
            word = MyceliumWord(
                phonetic_pattern=phonetic_pattern,
                chemical_signature=chemical_signature,
                electrical_signature=electrical_signature,
                meaning_concept=meaning_concept,
                context_cluster=group_name,
                formation_signals=[]  # Would contain original signals
            )

            words.append(word)
            self.mycelium_words.append(word)

        return words

    def _group_tokens_semantically(self,
                                   tokens: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group tokens by semantic similarity.

        Groups by token type and semantic weight:
        - High intensity (>0.7)
        - Medium intensity (>0.4)
        - Low intensity (≤0.4)

        Args:
            tokens: List of linguistic token dictionaries

        Returns:
            Dictionary mapping group keys to token lists
        """
        groups = defaultdict(list)

        for token in tokens:
            # Group by token type and semantic weight
            token_type = token.get('type', 'unknown')
            semantic_weight = token.get('semantic_weight', 0.5)

            if semantic_weight > 0.7:
                group_key = f"{token_type}_high_intensity"
            elif semantic_weight > 0.4:
                group_key = f"{token_type}_medium_intensity"
            else:
                group_key = f"{token_type}_low_intensity"

            groups[group_key].append(token)

        return dict(groups)

    async def assemble_sentences(self,
                                words: List[MyceliumWord],
                                structure: Dict[str, Any]) -> List[MyceliumSentence]:
        """
        Assemble words into mycelium sentences.

        Creates sentences by:
        - Grouping words semantically
        - Ordering words by network topology
        - Determining consciousness levels
        - Generating semantic flow

        Args:
            words: List of MyceliumWord objects
            structure: Syntactic structure dictionary

        Returns:
            List of MyceliumSentence objects
        """
        sentences = []

        if not words:
            return sentences

        word_order = structure.get('word_order', 'linear')
        phrase_structure = structure.get('phrase_structure', 'simple')
        semantic_relations = structure.get('semantic_relations', {})
        temporal_flow = structure.get('temporal_flow', 'steady')

        # Group words into sentence units
        sentence_groups = self._group_words_into_sentences(words, semantic_relations)

        for group in sentence_groups:
            # Order words according to network topology
            ordered_words = self._order_words_in_sentence(group, word_order)

            # Determine consciousness level of sentence
            sentence_consciousness = self._determine_sentence_consciousness(ordered_words)

            # Create sentence structure
            sentence = MyceliumSentence(
                words=ordered_words,
                syntactic_structure=f"{word_order}_{phrase_structure}",
                semantic_flow=self._generate_semantic_flow(ordered_words, semantic_relations),
                network_topology=phrase_structure,
                temporal_pattern=temporal_flow,
                consciousness_level=sentence_consciousness
            )

            sentences.append(sentence)
            self.mycelium_sentences.append(sentence)

        return sentences

    def _group_words_into_sentences(self,
                                    words: List[MyceliumWord],
                                    relations: Dict[str, List[str]]) -> List[List[MyceliumWord]]:
        """
        Group words into sentence units based on semantic relations.

        Groups by context cluster, ensuring minimum sentence size of 2 words.

        Args:
            words: List of MyceliumWord objects
            relations: Semantic relations dictionary

        Returns:
            List of word groups for sentences
        """
        if not words:
            return []

        # Simple grouping strategy - group by context cluster
        clusters = defaultdict(list)
        for word in words:
            clusters[word.context_cluster].append(word)

        # Convert to list of groups, ensuring minimum sentence size
        groups = []
        for cluster_words in clusters.values():
            if len(cluster_words) >= 2:
                groups.append(cluster_words)
            elif groups:  # Add single words to existing groups
                groups[-1].extend(cluster_words)
            else:  # Create new group if no existing groups
                groups.append(cluster_words)

        return groups

    def _order_words_in_sentence(self,
                                 words: List[MyceliumWord],
                                 word_order: str) -> List[MyceliumWord]:
        """
        Order words in sentence based on network topology rules.

        Word order patterns:
        - hub-spoke-peripheral: by electrical signature (highest first)
        - source-pathway-destination: by chemical complexity
        - gradient-diffusion-response: by concept complexity
        - default: preserve original order

        Args:
            words: List of MyceliumWord objects
            word_order: Word order pattern string

        Returns:
            Sorted list of MyceliumWord objects
        """
        if word_order == 'hub-spoke-peripheral':
            # Sort by electrical signature (hub = highest frequency)
            return sorted(words, key=lambda w: w.electrical_signature, reverse=True)

        elif word_order == 'source-pathway-destination':
            # Sort by chemical complexity (source = highest complexity)
            return sorted(words, key=lambda w: len(w.chemical_signature), reverse=True)

        elif word_order == 'gradient-diffusion-response':
            # Sort by meaning concept complexity
            complexity_order = ['sensing', 'signaling', 'processing', 'deciding', 'transcendent']

            def complexity_score(word):
                for i, concept_type in enumerate(complexity_order):
                    if concept_type in word.meaning_concept:
                        return i
                return 0

            return sorted(words, key=complexity_score)

        else:
            # Default: preserve original order
            return words

    def _determine_sentence_consciousness(self,
                                         words: List[MyceliumWord]) -> str:
        """
        Determine consciousness level of a sentence.

        Analyzes meaning concepts to assign consciousness level:
        - mycelial_metacognition: transcendent concepts
        - collective_consciousness: collective/group concepts
        - network_cognition: processing/deciding concepts
        - chemical_intelligence: signaling/communicating concepts
        - basic_awareness: other concepts

        Args:
            words: List of MyceliumWord objects

        Returns:
            Consciousness level string
        """
        if not words:
            return 'basic_awareness'

        # Analyze meaning concepts in the sentence
        concepts = [word.meaning_concept for word in words]

        if any('transcendent' in concept for concept in concepts):
            return 'mycelial_metacognition'
        elif any('collective' in concept or 'group' in concept for concept in concepts):
            return 'collective_consciousness'
        elif any('processing' in concept or 'deciding' in concept for concept in concepts):
            return 'network_cognition'
        elif any('signaling' in concept or 'communicating' in concept for concept in concepts):
            return 'chemical_intelligence'
        else:
            return 'basic_awareness'

    def _generate_semantic_flow(self,
                                words: List[MyceliumWord],
                                relations: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Generate semantic flow for sentence.

        Tracks:
        - Primary concept
        - Concept progression
        - Chemical compound flow
        - Electrical signature flow
        - Overall semantic coherence

        Args:
            words: List of MyceliumWord objects
            relations: Semantic relations dictionary

        Returns:
            Semantic flow dictionary
        """
        return {
            'primary_concept': words[0].meaning_concept if words else 'unknown',
            'concept_progression': [w.meaning_concept for w in words],
            'chemical_flow': self._trace_chemical_flow(words),
            'electrical_flow': self._trace_electrical_flow(words),
            'semantic_coherence': self._calculate_sentence_coherence(words)
        }

    def _trace_chemical_flow(self,
                            words: List[MyceliumWord]) -> Dict[str, List[float]]:
        """
        Trace chemical compound flow through sentence.

        Args:
            words: List of MyceliumWord objects

        Returns:
            Dictionary mapping compounds to concentration sequences
        """
        flow = defaultdict(list)

        for word in words:
            for compound, concentration in word.chemical_signature.items():
                flow[compound].append(concentration)

        return dict(flow)

    def _trace_electrical_flow(self,
                              words: List[MyceliumWord]) -> List[float]:
        """
        Trace electrical signature flow through sentence.

        Args:
            words: List of MyceliumWord objects

        Returns:
            List of electrical frequencies
        """
        return [word.electrical_signature for word in words]

    def _calculate_sentence_coherence(self,
                                     words: List[MyceliumWord]) -> float:
        """
        Calculate semantic coherence of sentence.

        Measures average similarity between adjacent words.

        Args:
            words: List of MyceliumWord objects

        Returns:
            Coherence score (0.0-1.0)
        """
        if len(words) < 2:
            return 1.0

        coherence_scores = []
        for i in range(len(words) - 1):
            similarity = self.pattern_analyzer.calculate_semantic_similarity(
                words[i],
                words[i + 1]
            )
            coherence_scores.append(similarity)

        if HAS_NUMPY and coherence_scores:
            return float(np.mean(coherence_scores))
        elif coherence_scores:
            return sum(coherence_scores) / len(coherence_scores)
        else:
            return 0.0


__all__ = ['LanguageSynthesizer']
