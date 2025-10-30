"""
Pattern Analyzer for Mycelium Language Generator

Analyzes syntactic patterns, semantic relationships, and linguistic structures
in mycelium language.
"""

from collections import defaultdict
from typing import Dict, List, Any

from .data_models import MyceliumWord
from .network_processor import NetworkProcessor


class PatternAnalyzer:
    """
    Analyzes patterns in mycelium language generation.

    Provides:
    - Syntactic structure generation
    - Semantic relationship mapping
    - Word similarity calculations
    - Chemical affinity analysis
    """

    def __init__(self, network_processor: NetworkProcessor):
        """
        Initialize pattern analyzer.

        Args:
            network_processor: NetworkProcessor instance for topology-based analysis
        """
        self.network = network_processor

    async def generate_syntactic_structure(self,
                                          words: List[MyceliumWord]) -> Dict[str, Any]:
        """
        Generate syntactic structure from network topology.

        Maps network topology to linguistic structure:
        - Word order based on clustering
        - Phrase structure based on fractal dimension
        - Semantic relations between words
        - Temporal flow from growth patterns

        Args:
            words: List of MyceliumWord objects

        Returns:
            Dictionary describing syntactic structure
        """
        structure = {
            'word_order': self.network.determine_word_order(),
            'phrase_structure': self.network.determine_phrase_structure(),
            'semantic_relations': self._map_semantic_relations(words),
            'temporal_flow': self.network.determine_temporal_flow()
        }

        return structure

    def _map_semantic_relations(self,
                                words: List[MyceliumWord]) -> Dict[str, List[str]]:
        """
        Map semantic relations between words.

        Analyzes:
        - Semantic similarity (meaning concepts)
        - Chemical affinity (compound similarities)

        Args:
            words: List of MyceliumWord objects

        Returns:
            Dictionary of semantic relation types and word pairs
        """
        relations = defaultdict(list)

        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words[i+1:], i+1):
                # Calculate semantic similarity
                similarity = self.calculate_semantic_similarity(word1, word2)

                if similarity > 0.7:
                    relations['high_similarity'].append(
                        f"{word1.phonetic_pattern} <-> {word2.phonetic_pattern}"
                    )
                elif similarity > 0.4:
                    relations['medium_similarity'].append(
                        f"{word1.phonetic_pattern} -> {word2.phonetic_pattern}"
                    )

                # Chemical affinity relations
                chemical_affinity = self.calculate_chemical_affinity(word1, word2)
                if chemical_affinity > 0.6:
                    relations['chemical_bond'].append(
                        f"{word1.phonetic_pattern} <=> {word2.phonetic_pattern}"
                    )

        return dict(relations)

    def calculate_semantic_similarity(self,
                                     word1: MyceliumWord,
                                     word2: MyceliumWord) -> float:
        """
        Calculate semantic similarity between two words.

        Considers:
        - Meaning concept similarity (50% weight)
        - Context cluster similarity (30% weight)
        - Electrical signature similarity (20% weight)

        Args:
            word1: First MyceliumWord
            word2: Second MyceliumWord

        Returns:
            Similarity score (0.0-1.0)
        """
        # Compare meaning concepts
        concept_similarity = 0.5 if word1.meaning_concept == word2.meaning_concept else 0.0

        # Compare context clusters
        cluster_similarity = 0.3 if word1.context_cluster == word2.context_cluster else 0.0

        # Compare electrical signatures
        electrical_diff = abs(word1.electrical_signature - word2.electrical_signature)
        electrical_similarity = max(0.0, 0.2 - electrical_diff / 10.0)

        return concept_similarity + cluster_similarity + electrical_similarity

    def calculate_chemical_affinity(self,
                                   word1: MyceliumWord,
                                   word2: MyceliumWord) -> float:
        """
        Calculate chemical affinity between two words.

        Measures similarity of chemical compound concentrations.
        Higher affinity when concentrations are similar.

        Args:
            word1: First MyceliumWord
            word2: Second MyceliumWord

        Returns:
            Affinity score (0.0-1.0)
        """
        affinity = 0.0
        common_compounds = (
            set(word1.chemical_signature.keys()) &
            set(word2.chemical_signature.keys())
        )

        for compound in common_compounds:
            conc1 = word1.chemical_signature[compound]
            conc2 = word2.chemical_signature[compound]
            # Higher affinity when concentrations are similar
            compound_affinity = 1.0 - abs(conc1 - conc2)
            affinity += compound_affinity

        # Normalize by number of compounds
        total_compounds = len(
            set(word1.chemical_signature.keys()) |
            set(word2.chemical_signature.keys())
        )
        return affinity / total_compounds if total_compounds > 0 else 0.0

    def analyze_word_complexity(self, word: MyceliumWord) -> float:
        """
        Analyze the complexity of a word.

        Considers:
        - Number of chemical compounds
        - Electrical signature magnitude
        - Phonetic pattern complexity

        Args:
            word: MyceliumWord to analyze

        Returns:
            Complexity score (0.0-1.0)
        """
        # Chemical complexity
        chemical_complexity = len(word.chemical_signature) / 20.0  # Normalize by max expected

        # Electrical complexity
        electrical_complexity = min(word.electrical_signature / 15.0, 1.0)

        # Phonetic complexity (number of phoneme segments)
        phonetic_segments = word.phonetic_pattern.count('-') + 1
        phonetic_complexity = min(phonetic_segments / 5.0, 1.0)

        # Average all complexities
        return (chemical_complexity + electrical_complexity + phonetic_complexity) / 3.0

    def detect_pattern_clusters(self,
                               words: List[MyceliumWord]) -> List[List[MyceliumWord]]:
        """
        Detect clusters of similar words.

        Groups words by:
        - Context cluster
        - Semantic similarity
        - Chemical affinity

        Args:
            words: List of MyceliumWord objects

        Returns:
            List of word clusters
        """
        if not words:
            return []

        # Group by context cluster first
        clusters_dict = defaultdict(list)
        for word in words:
            clusters_dict[word.context_cluster].append(word)

        # Convert to list of clusters
        clusters = list(clusters_dict.values())

        # Further refine by semantic similarity within clusters
        refined_clusters = []
        for cluster in clusters:
            if len(cluster) <= 2:
                refined_clusters.append(cluster)
            else:
                # Split large clusters by semantic similarity
                subcluster = [cluster[0]]
                for word in cluster[1:]:
                    # Check similarity with any word in current subcluster
                    similar = any(
                        self.calculate_semantic_similarity(word, w) > 0.6
                        for w in subcluster
                    )
                    if similar:
                        subcluster.append(word)
                    else:
                        # Start new subcluster
                        refined_clusters.append(subcluster)
                        subcluster = [word]

                if subcluster:
                    refined_clusters.append(subcluster)

        return refined_clusters

    def measure_linguistic_coherence(self, words: List[MyceliumWord]) -> float:
        """
        Measure overall linguistic coherence of a word sequence.

        Calculates average semantic similarity between adjacent words.

        Args:
            words: List of MyceliumWord objects

        Returns:
            Coherence score (0.0-1.0)
        """
        if len(words) < 2:
            return 1.0

        coherence_scores = []
        for i in range(len(words) - 1):
            similarity = self.calculate_semantic_similarity(words[i], words[i + 1])
            coherence_scores.append(similarity)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0


__all__ = ['PatternAnalyzer']
