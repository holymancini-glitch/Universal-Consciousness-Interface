"""
Vocabulary Manager for Mycelium Language Generator

Manages phonetic libraries, chemical vocabularies, syntactic rules, and
communication patterns used in mycelium language generation.
"""

import random
from typing import Dict, List, Any

from .data_models import MyceliumCommunicationType


class VocabularyManager:
    """
    Manages all vocabulary resources for mycelium language generation.

    This class initializes and maintains:
    - Phonetic libraries (sound patterns)
    - Chemical vocabularies (compound signatures)
    - Syntactic rules (grammar patterns)
    - Communication patterns (mycelium signal templates)
    """

    def __init__(self):
        """
        Initialize vocabulary manager with all language resources.
        """
        self.phonetic_library: Dict[str, str] = self._initialize_phonetic_library()
        self.chemical_vocabulary: Dict[str, Dict[str, float]] = self._initialize_chemical_vocabulary()
        self.syntactic_rules: Dict[str, List[str]] = self._initialize_syntactic_rules()
        self.communication_patterns: Dict[str, Any] = self._initialize_communication_patterns()

    def _initialize_communication_patterns(self) -> Dict[str, Any]:
        """
        Initialize mycelium communication patterns based on research.

        Creates patterns for:
        - Chemical gradient patterns (60+ documented chemical signals)
        - Electrical pulse patterns (50+ documented by Adamatzky research)
        - Nutrient flow patterns (resource allocation intelligence)
        - Network resonance patterns (collective consciousness)

        Returns:
            Dictionary of communication patterns with their characteristics
        """
        patterns = {}

        # Chemical gradient patterns (60+ documented chemical signals)
        for i in range(1, 61):
            patterns[f"chemical_pattern_{i}"] = {
                'type': MyceliumCommunicationType.CHEMICAL_GRADIENT,
                'primary_compound': f'compound_{i}',
                'concentration_gradient': random.uniform(0.1, 1.0),
                'diffusion_rate': random.uniform(0.01, 0.1),
                'meaning_category': random.choice([
                    'resource_sharing',
                    'threat_warning',
                    'growth_coordination',
                    'network_expansion'
                ])
            }

        # Electrical pulse patterns (50+ documented by Adamatzky research)
        for i in range(1, 51):
            patterns[f"electrical_pattern_{i}"] = {
                'type': MyceliumCommunicationType.ELECTRICAL_PULSE,
                'frequency': 0.1 + (i * 0.05),  # Hz
                'amplitude': 0.2 + (i * 0.02),
                'pulse_duration': 1.0 + (i * 0.1),
                'meaning_category': random.choice([
                    'information_relay',
                    'decision_propagation',
                    'network_synchronization',
                    'collective_processing'
                ])
            }

        # Nutrient flow patterns (resource allocation intelligence)
        for i in range(1, 31):
            patterns[f"nutrient_pattern_{i}"] = {
                'type': MyceliumCommunicationType.NUTRIENT_FLOW,
                'flow_rate': random.uniform(0.1, 2.0),
                'resource_type': random.choice([
                    'carbon',
                    'nitrogen',
                    'phosphorus',
                    'water',
                    'minerals'
                ]),
                'allocation_strategy': random.choice([
                    'optimal_distribution',
                    'priority_routing',
                    'emergency_reallocation'
                ]),
                'meaning_category': 'resource_intelligence'
            }

        # Network resonance patterns (collective consciousness)
        for i in range(1, 21):
            patterns[f"resonance_pattern_{i}"] = {
                'type': MyceliumCommunicationType.NETWORK_RESONANCE,
                'resonance_frequency': random.uniform(1.0, 10.0),
                'network_coverage': random.uniform(0.3, 1.0),
                'synchronization_level': random.uniform(0.5, 0.95),
                'meaning_category': 'collective_consciousness'
            }

        return patterns

    def _initialize_phonetic_library(self) -> Dict[str, str]:
        """
        Initialize phonetic patterns based on mycelium signal characteristics.

        Creates phonemes inspired by:
        - Chemical processes (soft, flowing sounds)
        - Electrical signals (sharp, rhythmic sounds)
        - Network structures (complex, interconnected sounds)
        - Consciousness (deep, resonant sounds)

        Returns:
            Dictionary mapping phoneme IDs to phonetic patterns
        """
        phonetics = {}

        # Chemical-inspired phonemes (soft, flowing sounds)
        chemical_phonemes = [
            'myu', 'cel', 'thy', 'fim', 'spor', 'hyph', 'mel', 'enz',
            'dif', 'gra', 'con', 'flu', 'bio', 'sym', 'net', 'web'
        ]

        # Electrical-inspired phonemes (sharp, rhythmic sounds)
        electrical_phonemes = [
            'zik', 'puls', 'amp', 'freq', 'volt', 'sync', 'res', 'osc',
            'sig', 'wave', 'curr', 'ion', 'flux', 'char', 'pot', 'field'
        ]

        # Network-inspired phonemes (complex, interconnected sounds)
        network_phonemes = [
            'nod', 'link', 'hub', 'path', 'conn', 'mesh', 'grid', 'tree',
            'loop', 'flow', 'span', 'edge', 'vert', 'clust', 'dist', 'cent'
        ]

        # Consciousness-inspired phonemes (deep, resonant sounds)
        consciousness_phonemes = [
            'awa', 'cog', 'mind', 'know', 'feel', 'sens', 'per', 'con',
            'meta', 'self', 'ref', 'rec', 'mem', 'learn', 'adapt', 'evolv'
        ]

        all_phonemes = (chemical_phonemes + electrical_phonemes +
                       network_phonemes + consciousness_phonemes)

        for i, phoneme in enumerate(all_phonemes):
            phonetics[f"phoneme_{i}"] = phoneme

        return phonetics

    def _initialize_chemical_vocabulary(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize chemical compound signatures for language elements.

        Creates vocabulary entries combining:
        - Basic chemical components found in fungi
        - Consciousness-affecting compounds (e.g., muscimol from Amanita muscaria)
        - Signaling molecules

        Returns:
            Dictionary of chemical vocabulary entries with compound concentrations
        """
        vocabulary = {}

        # Basic chemical components found in fungi
        base_compounds = [
            'chitin', 'glucan', 'melanin', 'ergosterol', 'trehalose',
            'glycerol', 'mannitol', 'arabitol', 'enzyme_complex', 'neurotransmitter'
        ]

        # Consciousness-affecting compounds (like muscimol from Amanita muscaria)
        consciousness_compounds = [
            'muscimol', 'ibotenic_acid', 'psilocybin', 'psilocin',
            'tryptamine', 'serotonin', 'dopamine', 'acetylcholine'
        ]

        # Signaling molecules
        signaling_compounds = [
            'cyclic_amp', 'calcium_ion', 'nitric_oxide', 'hydrogen_peroxide',
            'volatile_organic', 'peptide_signal', 'hormone_analog'
        ]

        all_compounds = base_compounds + consciousness_compounds + signaling_compounds

        for i in range(100):  # Create 100 chemical vocabulary entries
            compound_signature = {}
            selected_compounds = random.sample(all_compounds, random.randint(3, 7))

            for compound in selected_compounds:
                compound_signature[compound] = random.uniform(0.1, 1.0)

            vocabulary[f"chemical_vocab_{i}"] = compound_signature

        return vocabulary

    def _initialize_syntactic_rules(self) -> Dict[str, List[str]]:
        """
        Initialize syntactic rules based on network topology patterns.

        Rules are inspired by:
        - Hyphal growth patterns (word formation)
        - Network communication patterns (sentence structure)
        - Resource sharing patterns (semantic relationships)
        - Growth and communication timing (temporal patterns)

        Returns:
            Dictionary of syntactic rule categories
        """
        rules = {
            # Basic word formation (hyphal growth patterns)
            'word_formation': [
                'root + extension',
                'branching + merger',
                'node + connection',
                'cluster + expansion'
            ],

            # Sentence structure (network communication patterns)
            'sentence_structure': [
                'source → pathway → destination',
                'hub → spoke → peripheral',
                'gradient → flow → accumulation',
                'signal → amplification → response'
            ],

            # Semantic relationships (resource sharing patterns)
            'semantic_relations': [
                'mutual_benefit',
                'resource_exchange',
                'information_sharing',
                'collective_decision',
                'network_optimization'
            ],

            # Temporal patterns (growth and communication timing)
            'temporal_patterns': [
                'immediate_response',
                'gradual_buildup',
                'rhythmic_oscillation',
                'burst_communication',
                'sustained_flow'
            ]
        }

        return rules

    def get_random_phoneme(self) -> str:
        """
        Get a random phoneme from the library.

        Returns:
            Random phonetic pattern
        """
        return random.choice(list(self.phonetic_library.values()))

    def get_chemical_vocab(self, key: str) -> Dict[str, float]:
        """
        Get a chemical vocabulary entry by key.

        Args:
            key: The vocabulary entry key

        Returns:
            Dictionary of chemical compounds and concentrations
        """
        return self.chemical_vocabulary.get(key, {})


__all__ = ['VocabularyManager']
