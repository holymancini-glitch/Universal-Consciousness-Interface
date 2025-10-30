"""
Biochemical Translator for Mycelium Language Generator

Translates mycelium signals (chemical, electrical, nutrient) into phonetic
patterns and linguistic tokens.
"""

import random
from typing import Dict, List, Any
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .data_models import MyceliumSignal, MyceliumCommunicationType
from .vocabulary_manager import VocabularyManager


class BiochemicalTranslator:
    """
    Translates biochemical and electrical signals into linguistic elements.

    Converts mycelium communication signals into:
    - Phonetic patterns (sound representations)
    - Chemical signatures (compound profiles)
    - Electrical signatures (frequency patterns)
    - Meaning concepts (semantic content)
    """

    def __init__(self, vocabulary_manager: VocabularyManager):
        """
        Initialize biochemical translator.

        Args:
            vocabulary_manager: VocabularyManager instance for phonetic/chemical vocab
        """
        self.vocabulary = vocabulary_manager

    async def process_signals_to_tokens(self,
                                       signals: List[MyceliumSignal]) -> List[Dict[str, Any]]:
        """
        Convert mycelium signals into linguistic tokens.

        Each signal type is processed differently:
        - Chemical gradients → phonetic roots from compounds
        - Electrical pulses → frequency-based patterns
        - Nutrient flows → flow-based phonetics
        - Network resonance → collective patterns

        Args:
            signals: List of MyceliumSignal objects

        Returns:
            List of linguistic token dictionaries
        """
        tokens = []

        for signal in signals:
            # Extract linguistic features from each signal type
            if signal.signal_type == MyceliumCommunicationType.CHEMICAL_GRADIENT:
                token = {
                    'type': 'chemical_token',
                    'phonetic_root': self._chemical_to_phonetic(signal.chemical_composition),
                    'semantic_weight': signal.intensity,
                    'temporal_signature': signal.duration,
                    'spatial_pattern': signal.spatial_pattern
                }

            elif signal.signal_type == MyceliumCommunicationType.ELECTRICAL_PULSE:
                token = {
                    'type': 'electrical_token',
                    'phonetic_root': self._frequency_to_phonetic(signal.electrical_frequency),
                    'semantic_weight': signal.intensity,
                    'rhythm_pattern': signal.duration,
                    'network_resonance': signal.electrical_frequency
                }

            elif signal.signal_type == MyceliumCommunicationType.NUTRIENT_FLOW:
                token = {
                    'type': 'flow_token',
                    'phonetic_root': self._flow_to_phonetic(signal),
                    'semantic_weight': signal.intensity,
                    'direction_pattern': signal.spatial_pattern,
                    'resource_signature': signal.chemical_composition
                }

            elif signal.signal_type == MyceliumCommunicationType.NETWORK_RESONANCE:
                token = {
                    'type': 'resonance_token',
                    'phonetic_root': self._resonance_to_phonetic(signal),
                    'semantic_weight': signal.intensity,
                    'collective_pattern': signal.spatial_pattern,
                    'consciousness_marker': signal.electrical_frequency
                }

            else:
                # Generic processing for other signal types
                token = {
                    'type': 'generic_token',
                    'phonetic_root': self._generic_to_phonetic(signal),
                    'semantic_weight': signal.intensity,
                    'pattern_signature': signal.spatial_pattern
                }

            tokens.append(token)

        return tokens

    def _chemical_to_phonetic(self, chemical_composition: Dict[str, float]) -> str:
        """
        Convert chemical composition to phonetic pattern.

        Maps specific compounds to characteristic sounds:
        - Melanin → 'mel'
        - Chitin → 'chi'
        - Enzyme → 'enz'
        - High concentration → 'amp' (amplified)
        - Low concentration → 'sub' (subtle)

        Args:
            chemical_composition: Dictionary of compounds and concentrations

        Returns:
            Phonetic pattern string
        """
        phonetic_elements = []

        for compound, concentration in chemical_composition.items():
            # Map chemical properties to sound characteristics
            if 'melanin' in compound:
                phonetic_elements.append('mel')
            elif 'chitin' in compound:
                phonetic_elements.append('chi')
            elif 'enzyme' in compound:
                phonetic_elements.append('enz')
            elif concentration > 0.8:
                phonetic_elements.append('amp')  # High concentration = amplified sound
            elif concentration < 0.2:
                phonetic_elements.append('sub')  # Low concentration = subtle sound

        if not phonetic_elements:
            phonetic_elements = [self.vocabulary.get_random_phoneme()]

        return '-'.join(phonetic_elements[:3])  # Combine up to 3 elements

    def _frequency_to_phonetic(self, frequency: float) -> str:
        """
        Convert electrical frequency to phonetic pattern.

        Frequency ranges:
        - <1.0 Hz: low-freq-puls
        - 1.0-5.0 Hz: mid-freq-wave
        - 5.0-10.0 Hz: high-freq-osc
        - >10.0 Hz: ultra-freq-burst

        Args:
            frequency: Electrical frequency in Hz

        Returns:
            Phonetic pattern string
        """
        if frequency < 1.0:
            return 'low-freq-puls'
        elif frequency < 5.0:
            return 'mid-freq-wave'
        elif frequency < 10.0:
            return 'high-freq-osc'
        else:
            return 'ultra-freq-burst'

    def _flow_to_phonetic(self, signal: MyceliumSignal) -> str:
        """
        Convert nutrient flow to phonetic pattern.

        Combines flow intensity with spatial pattern:
        - High intensity (>0.8): 'rush'
        - Medium intensity (>0.5): 'flow'
        - Low intensity: 'trickle'
        Combined with: radial, direct, or diffuse

        Args:
            signal: MyceliumSignal with flow information

        Returns:
            Phonetic pattern string
        """
        flow_intensity = signal.intensity
        spatial_pattern = signal.spatial_pattern

        if flow_intensity > 0.8:
            base = 'rush'
        elif flow_intensity > 0.5:
            base = 'flow'
        else:
            base = 'trickle'

        if 'radial' in spatial_pattern:
            return f'{base}-radial'
        elif 'directional' in spatial_pattern:
            return f'{base}-direct'
        else:
            return f'{base}-diffuse'

    def _resonance_to_phonetic(self, signal: MyceliumSignal) -> str:
        """
        Convert network resonance to phonetic pattern.

        Combines resonance intensity with frequency:
        - High intensity (>0.5): 'res'
        - Low intensity: 'sub-res'
        - High frequency (>5.0): harmonic
        - Low frequency: fundamental

        Args:
            signal: MyceliumSignal with resonance information

        Returns:
            Phonetic pattern string
        """
        resonance_freq = signal.electrical_frequency
        intensity = signal.intensity

        base_sound = 'res' if intensity > 0.5 else 'sub-res'

        if resonance_freq > 5.0:
            return f'{base_sound}-harmonic'
        else:
            return f'{base_sound}-fundamental'

    def _generic_to_phonetic(self, signal: MyceliumSignal) -> str:
        """
        Generic signal to phonetic conversion.

        Args:
            signal: Any MyceliumSignal

        Returns:
            Random phonetic pattern from vocabulary
        """
        return self.vocabulary.get_random_phoneme()

    def combine_phonetic_patterns(self, patterns: List[str]) -> str:
        """
        Combine multiple phonetic patterns into a word.

        Takes up to 3 patterns and joins them with hyphens.

        Args:
            patterns: List of phonetic pattern strings

        Returns:
            Combined phonetic pattern
        """
        if not patterns:
            return self.vocabulary.get_random_phoneme()

        # Take up to 3 patterns and combine them
        selected_patterns = patterns[:3]
        return '-'.join(selected_patterns)

    def generate_chemical_signature(self,
                                   token_group: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Generate chemical signature for a word from token group.

        Combines base chemical vocabulary with token-specific modifiers:
        - Chemical tokens amplify compounds
        - Resonance tokens add neurotransmitters

        Args:
            token_group: List of token dictionaries

        Returns:
            Dictionary of chemical compounds and concentrations
        """
        signature = {}

        # Base signature from random chemical vocabulary
        base_vocab = random.choice(list(self.vocabulary.chemical_vocabulary.values()))
        signature.update(base_vocab)

        # Modify based on token characteristics
        for token in token_group:
            if token.get('type') == 'chemical_token':
                # Amplify chemical compounds
                for compound in signature:
                    signature[compound] *= 1.2
            elif token.get('type') == 'resonance_token':
                # Add consciousness compounds
                signature['neurotransmitter'] = signature.get('neurotransmitter', 0) + 0.3

        # Normalize values
        max_val = max(signature.values()) if signature else 1.0
        for compound in signature:
            signature[compound] = min(signature[compound] / max_val, 1.0)

        return signature

    def calculate_electrical_signature(self,
                                      token_group: List[Dict[str, Any]]) -> float:
        """
        Calculate electrical signature for a word from token group.

        Averages electrical frequencies from tokens with resonance data.

        Args:
            token_group: List of token dictionaries

        Returns:
            Average electrical frequency (Hz)
        """
        electrical_values = []

        for token in token_group:
            if 'network_resonance' in token:
                electrical_values.append(token['network_resonance'])
            elif 'consciousness_marker' in token:
                electrical_values.append(token['consciousness_marker'])
            else:
                electrical_values.append(random.uniform(0.1, 10.0))

        if HAS_NUMPY and electrical_values:
            return float(np.mean(electrical_values))
        elif electrical_values:
            return sum(electrical_values) / len(electrical_values)
        else:
            return 1.0

    def derive_meaning_concept(self,
                              token_group: List[Dict[str, Any]],
                              consciousness_level: str) -> str:
        """
        Derive meaning concept based on consciousness level and tokens.

        Consciousness levels:
        - basic_awareness: sensing, detecting, responding, growing
        - chemical_intelligence: signaling, communicating, sharing, coordinating
        - network_cognition: processing, deciding, optimizing, adapting
        - collective_consciousness: collective-thinking, group-deciding, distributed-intelligence
        - mycelial_metacognition: self-awareness, meta-processing, consciousness-recursion

        Args:
            token_group: List of token dictionaries
            consciousness_level: Consciousness level string

        Returns:
            Meaning concept string
        """
        concepts = {
            'basic_awareness': ['sensing', 'detecting', 'responding', 'growing'],
            'chemical_intelligence': ['signaling', 'communicating', 'sharing', 'coordinating'],
            'network_cognition': ['processing', 'deciding', 'optimizing', 'adapting'],
            'collective_consciousness': [
                'collective-thinking',
                'group-deciding',
                'network-awareness',
                'distributed-intelligence'
            ],
            'mycelial_metacognition': [
                'self-awareness',
                'meta-processing',
                'consciousness-recursion',
                'transcendent-understanding'
            ]
        }

        level_concepts = concepts.get(consciousness_level, concepts['network_cognition'])

        # Determine dominant token type
        token_types = [t.get('type', 'unknown') for t in token_group]
        if not token_types:
            return random.choice(level_concepts)

        dominant_type = max(set(token_types), key=token_types.count)

        # Prefix concept with token type
        base_concept = random.choice(level_concepts)
        if dominant_type == 'chemical_token':
            return f"chemical-{base_concept}"
        elif dominant_type == 'electrical_token':
            return f"electrical-{base_concept}"
        elif dominant_type == 'resonance_token':
            return f"resonance-{base_concept}"
        else:
            return base_concept


__all__ = ['BiochemicalTranslator']
