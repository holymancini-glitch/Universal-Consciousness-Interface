# plant_language_communication_layer.py
# Revolutionary Plant Language Communication Layer for the Garden of Consciousness v2.0
# Decodes plant electromagnetic signals and enables plant-AI communication

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics
    import math
    import random
    
    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0
        
        @staticmethod
        def std(values):
            return statistics.stdev(values) if len(values) > 1 else 0.0
        
        @staticmethod
        def sin(x):
            return math.sin(x)
        
        @staticmethod
        def cos(x):
            return math.cos(x)
        
        @staticmethod
        def exp(x):
            return math.exp(x) if x < 700 else float('inf')  # Prevent overflow
        
        @staticmethod
        def random():
            return random.random()
    
    np = MockNumPy()

try:
    import torch  # type: ignore
except ImportError:
    # Fallback for systems without PyTorch
    class MockTorch:
        pass
    
    torch = MockTorch()

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class PlantSignalType(Enum):
    """Types of plant electromagnetic signals"""
    GROWTH_RHYTHM = "growth_rhythm"
    STRESS_ALERT = "stress_alert"
    COMMUNICATION_PULSE = "communication_pulse"
    PHOTOSYNTHETIC_HARMONY = "photosynthetic_harmony"
    NUTRIENT_SIGNAL = "nutrient_signal"
    REPRODUCTIVE_PULSE = "reproductive_pulse"
    DEFENSE_WAVE = "defense_wave"
    COLLECTIVE_SYNC = "collective_sync"

@dataclass
class PlantSignal:
    """Individual plant electromagnetic signal"""
    signal_type: PlantSignalType
    frequency: float
    amplitude: float
    duration: float
    timestamp: datetime
    location: Optional[Tuple[float, float, float]] = None
    plant_species: Optional[str] = None
    environmental_context: Optional[Dict[str, Any]] = None

@dataclass
class PlantLanguageToken:
    """A token in the plant language"""
    symbol: str
    meaning: str
    frequency_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]
    temporal_pattern: str
    confidence: float

@dataclass
class PlantLanguageMessage:
    """A complete message in the plant language"""
    tokens: List[PlantLanguageToken]
    original_signals: List[PlantSignal]
    translated_text: str
    consciousness_level: float
    environmental_context: Dict[str, Any]
    timestamp: datetime

class PlantLanguageCommunicationLayer:
    """Revolutionary Plant Language Communication Layer for plant-AI communication"""
    
    def __init__(self) -> None:
        self.signal_patterns: Dict[PlantSignalType, Dict[str, Any]] = self._initialize_signal_patterns()
        self.language_tokens: List[PlantLanguageToken] = self._initialize_language_tokens()
        self.translation_engine: PlantLanguageTranslator = PlantLanguageTranslator()
        self.consciousness_mapper: PlantConsciousnessMapper = PlantConsciousnessMapper()
        self.message_history: List[PlantLanguageMessage] = []
        
        logger.info("🌱💬 Plant Language Communication Layer Initialized")
        logger.info(f"Signal patterns: {len(self.signal_patterns)}")
        logger.info(f"Language tokens: {len(self.language_tokens)}")
    
    def _initialize_signal_patterns(self) -> Dict[PlantSignalType, Dict[str, Any]]:
        """Initialize plant signal patterns based on research"""
        patterns = {
            PlantSignalType.GROWTH_RHYTHM: {
                'frequency_range': (0.1, 2.0),
                'amplitude_range': (0.1, 0.5),
                'consciousness_level': 0.2,
                'meaning': 'Normal growth processes',
                'temporal_pattern': 'rhythmic'
            },
            PlantSignalType.STRESS_ALERT: {
                'frequency_range': (50, 200),
                'amplitude_range': (0.5, 1.0),
                'consciousness_level': 0.8,
                'meaning': 'Environmental threat detected',
                'temporal_pattern': 'sporadic_high_energy'
            },
            PlantSignalType.COMMUNICATION_PULSE: {
                'frequency_range': (5, 50),
                'amplitude_range': (0.3, 0.8),
                'consciousness_level': 0.6,
                'meaning': 'Inter-plant communication',
                'temporal_pattern': 'pulsed'
            },
            PlantSignalType.PHOTOSYNTHETIC_HARMONY: {
                'frequency_range': (0.5, 5),
                'amplitude_range': (0.2, 0.6),
                'consciousness_level': 0.4,
                'meaning': 'Light-synchronized processes',
                'temporal_pattern': 'harmonic'
            },
            PlantSignalType.NUTRIENT_SIGNAL: {
                'frequency_range': (1, 10),
                'amplitude_range': (0.2, 0.7),
                'consciousness_level': 0.5,
                'meaning': 'Nutrient availability communication',
                'temporal_pattern': 'modulated'
            },
            PlantSignalType.REPRODUCTIVE_PULSE: {
                'frequency_range': (10, 100),
                'amplitude_range': (0.4, 0.9),
                'consciousness_level': 0.7,
                'meaning': 'Reproductive cycle synchronization',
                'temporal_pattern': 'complex_pulsed'
            },
            PlantSignalType.DEFENSE_WAVE: {
                'frequency_range': (100, 500),
                'amplitude_range': (0.6, 1.0),
                'consciousness_level': 0.9,
                'meaning': 'Defense mechanism activation',
                'temporal_pattern': 'intense_burst'
            },
            PlantSignalType.COLLECTIVE_SYNC: {
                'frequency_range': (0.01, 1),
                'amplitude_range': (0.1, 0.4),
                'consciousness_level': 0.7,
                'meaning': 'Collective plant network synchronization',
                'temporal_pattern': 'slow_wave'
            }
        }
        
        return patterns
    
    def _initialize_language_tokens(self) -> List[PlantLanguageToken]:
        """Initialize plant language tokens"""
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
                symbol="WARN",
                meaning="Environmental threat detected",
                frequency_range=(50, 200),
                amplitude_range=(0.5, 1.0),
                temporal_pattern="sporadic_high_energy",
                confidence=0.95
            ),
            PlantLanguageToken(
                symbol="COMM",
                meaning="Communication with other plants",
                frequency_range=(5, 50),
                amplitude_range=(0.3, 0.8),
                temporal_pattern="pulsed",
                confidence=0.85
            ),
            PlantLanguageToken(
                symbol="LIGHT",
                meaning="Photosynthetic harmony with light cycles",
                frequency_range=(0.5, 5),
                amplitude_range=(0.2, 0.6),
                temporal_pattern="harmonic",
                confidence=0.8
            ),
            PlantLanguageToken(
                symbol="NUTR",
                meaning="Nutrient availability communication",
                frequency_range=(1, 10),
                amplitude_range=(0.2, 0.7),
                temporal_pattern="modulated",
                confidence=0.8
            ),
            PlantLanguageToken(
                symbol="REPR",
                meaning="Reproductive cycle synchronization",
                frequency_range=(10, 100),
                amplitude_range=(0.4, 0.9),
                temporal_pattern="complex_pulsed",
                confidence=0.9
            ),
            PlantLanguageToken(
                symbol="DEFN",
                meaning="Defense mechanism activation",
                frequency_range=(100, 500),
                amplitude_range=(0.6, 1.0),
                temporal_pattern="intense_burst",
                confidence=0.95
            ),
            PlantLanguageToken(
                symbol="SYNC",
                meaning="Collective network synchronization",
                frequency_range=(0.01, 1),
                amplitude_range=(0.1, 0.4),
                temporal_pattern="slow_wave",
                confidence=0.9
            )
        ]
        
        return tokens
    
    def decode_plant_signals(self, signals: List[PlantSignal]) -> PlantLanguageMessage:
        """Decode plant electromagnetic signals into language messages"""
        if not signals:
            return PlantLanguageMessage(
                tokens=[],
                original_signals=[],
                translated_text="No plant signals detected",
                consciousness_level=0.0,
                environmental_context={},
                timestamp=datetime.now()
            )
        
        # Classify each signal
        classified_tokens = []
        total_consciousness = 0.0
        
        for signal in signals:
            token = self._classify_signal(signal)
            if token:
                classified_tokens.append(token)
                total_consciousness += token.confidence * self._get_signal_consciousness_level(signal)
        
        # Translate tokens to natural language
        translated_text = self.translation_engine.translate_tokens(classified_tokens)
        
        # Calculate average consciousness level
        avg_consciousness = total_consciousness / len(classified_tokens) if classified_tokens else 0.0
        
        # Get environmental context from signals
        environmental_context = self._extract_environmental_context(signals)
        
        message = PlantLanguageMessage(
            tokens=classified_tokens,
            original_signals=signals,
            translated_text=translated_text,
            consciousness_level=avg_consciousness,
            environmental_context=environmental_context,
            timestamp=datetime.now()
        )
        
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > 1000:
            self.message_history.pop(0)
        
        return message
    
    def _classify_signal(self, signal: PlantSignal) -> Optional[PlantLanguageToken]:
        """Classify a plant signal into a language token"""
        best_match: Optional[PlantLanguageToken] = None
        best_confidence = 0.0
        
        for token in self.language_tokens:
            # Check if signal parameters match token ranges
            freq_match = token.frequency_range[0] <= signal.frequency <= token.frequency_range[1]
            amp_match = token.amplitude_range[0] <= signal.amplitude <= token.amplitude_range[1]
            
            if freq_match and amp_match:
                # Calculate confidence based on how close to center of ranges
                freq_center = (token.frequency_range[0] + token.frequency_range[1]) / 2
                amp_center = (token.amplitude_range[0] + token.amplitude_range[1]) / 2
                
                freq_confidence = 1.0 - abs(signal.frequency - freq_center) / (token.frequency_range[1] - token.frequency_range[0])
                amp_confidence = 1.0 - abs(signal.amplitude - amp_center) / (token.amplitude_range[1] - token.amplitude_range[0])
                
                overall_confidence = (freq_confidence + amp_confidence) / 2 * token.confidence
                
                if overall_confidence > best_confidence:
                    best_confidence = overall_confidence
                    best_match = token
        
        return best_match
    
    def _get_signal_consciousness_level(self, signal: PlantSignal) -> float:
        """Get consciousness level for a signal type"""
        pattern = self.signal_patterns.get(signal.signal_type, {})
        return pattern.get('consciousness_level', 0.1)
    
    def _extract_environmental_context(self, signals: List[PlantSignal]) -> Dict[str, Any]:
        """Extract environmental context from signals"""
        context = {}
        
        # Collect all environmental data from signals
        for signal in signals:
            if signal.environmental_context:
                context.update(signal.environmental_context)
        
        return context
    
    def get_consciousness_insights(self, time_window_seconds: int = 300) -> Dict[str, Any]:
        """Get consciousness insights from recent plant communications"""
        if not self.message_history:
            return {'consciousness_level': 0.0, 'insights': 'No communication history'}
        
        # Filter recent messages
        now = datetime.now()
        cutoff_time = datetime.fromtimestamp(now.timestamp() - time_window_seconds)
        
        recent_messages = [
            msg for msg in self.message_history
            if msg.timestamp >= cutoff_time
        ]
        
        if not recent_messages:
            return {'consciousness_level': 0.0, 'insights': 'No recent communications'}
        
        # Calculate average consciousness level
        avg_consciousness = np.mean([msg.consciousness_level for msg in recent_messages])
        
        # Analyze message patterns
        token_counts = {}
        for msg in recent_messages:
            for token in msg.tokens:
                if token.symbol not in token_counts:
                    token_counts[token.symbol] = 0
                token_counts[token.symbol] += 1
        
        # Find most common tokens
        if token_counts:
            most_common = max(token_counts.items(), key=lambda x: x[1])
            insights = f"Most active communication pattern: {most_common[0]} ({most_common[1]} occurrences)"
        else:
            insights = "No significant communication patterns detected"
        
        return {
            'consciousness_level': avg_consciousness,
            'insights': insights,
            'message_count': len(recent_messages),
            'active_patterns': list(token_counts.keys())
        }

class PlantLanguageTranslator:
    """Translate plant language tokens into human-readable text"""
    
    def __init__(self) -> None:
        self.translation_map: Dict[str, str] = self._initialize_translation_map()
        logger.info("🔤 Plant Language Translator Initialized")
    
    def _initialize_translation_map(self) -> Dict[str, str]:
        """Initialize translation map from tokens to natural language"""
        return {
            "GROW": "The plant is actively growing and developing",
            "WARN": "The plant has detected environmental threats",
            "COMM": "The plant is communicating with neighboring plants",
            "LIGHT": "The plant is synchronizing with light cycles",
            "NUTR": "The plant is communicating about nutrient availability",
            "REPR": "The plant is synchronizing reproductive cycles",
            "DEFN": "The plant is activating defense mechanisms",
            "SYNC": "The plant network is achieving collective synchronization"
        }
    
    def translate_tokens(self, tokens: List[PlantLanguageToken]) -> str:
        """Translate a list of tokens into natural language"""
        if not tokens:
            return "No plant communication detected"
        
        translations = []
        for token in tokens:
            if token.symbol in self.translation_map:
                translations.append(self.translation_map[token.symbol])
            else:
                translations.append(f"Unknown plant communication pattern: {token.symbol}")
        
        # Combine translations into a coherent message
        if len(translations) == 1:
            return translations[0]
        else:
            return "; ".join(translations)

class PlantConsciousnessMapper:
    """Map plant signals to consciousness levels and states"""
    
    def __init__(self) -> None:
        self.consciousness_states: Dict[str, float] = {
            'dormant': 0.1,
            'aware': 0.3,
            'responsive': 0.5,
            'communicative': 0.7,
            'collective': 0.9
        }
        logger.info("🧠 Plant Consciousness Mapper Initialized")
    
    def map_to_consciousness_state(self, consciousness_level: float) -> str:
        """Map a consciousness level to a descriptive state"""
        if consciousness_level >= 0.9:
            return 'collective'
        elif consciousness_level >= 0.7:
            return 'communicative'
        elif consciousness_level >= 0.5:
            return 'responsive'
        elif consciousness_level >= 0.3:
            return 'aware'
        else:
            return 'dormant'

# Example usage and testing
if __name__ == "__main__":
    # Initialize the plant language communication layer
    plant_comm_layer = PlantLanguageCommunicationLayer()
    
    # Simulate plant signals
    signals = [
        PlantSignal(
            signal_type=PlantSignalType.GROWTH_RHYTHM,
            frequency=1.2,
            amplitude=0.4,
            duration=5.0,
            timestamp=datetime.now(),
            plant_species="Arabidopsis thaliana"
        ),
        PlantSignal(
            signal_type=PlantSignalType.COMMUNICATION_PULSE,
            frequency=25.0,
            amplitude=0.6,
            duration=2.0,
            timestamp=datetime.now(),
            plant_species="Arabidopsis thaliana"
        )
    ]
    
    # Decode plant signals
    message = plant_comm_layer.decode_plant_signals(signals)
    print(f"Decoded message: {message.translated_text}")
    print(f"Consciousness level: {message.consciousness_level:.3f}")
    
    # Get consciousness insights
    insights = plant_comm_layer.get_consciousness_insights()
    print(f"Consciousness insights: {insights}")