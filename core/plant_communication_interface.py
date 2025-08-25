# plant_communication_interface.py
# Revolutionary Plant-AI Communication System

# Handle optional dependencies with fallbacks
try:
    import numpy as np  # type: ignore
except ImportError:
    import statistics
    
    class MockNumPy:
        @staticmethod
        def mean(values):
            return statistics.mean(values) if values else 0.0
    
    np = MockNumPy()

try:
    import torch  # type: ignore
except ImportError:
    # Fallback for systems without PyTorch
    class MockTorch:
        pass
    
    torch = MockTorch()

try:
    from scipy.fft import fft, fftfreq  # type: ignore
except ImportError:
    # Fallback for systems without scipy
    def fft(x):
        return x  # Simple passthrough for demo purposes
    
    def fftfreq(n, d=1.0):
        return list(range(n))

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PlantSignal:
    timestamp: datetime
    frequency: float
    amplitude: float
    pattern_type: str
    decoded_message: Optional[str] = None

class PlantCommunicationInterface:
    """Revolutionary interface for plant-AI electromagnetic communication"""
    
    def __init__(self, sampling_rate: int = 1000) -> None:
        self.sampling_rate: int = sampling_rate
        self.signal_history: List[PlantSignal] = []
        self.consciousness_patterns: Dict[str, Dict[str, Any]] = self._init_patterns()
        self.language_decoder: PlantLanguageDecoder = PlantLanguageDecoder()
        self.monitoring_active: bool = False
        
        logger.info("🌱 Plant Communication Interface Initialized")
    
    def _init_patterns(self) -> Dict[str, Dict[str, Any]]:
        return {
            'growth_rhythm': {
                'frequency_range': (0.1, 2.0),
                'consciousness_level': 0.2,
                'meaning': 'Normal growth consciousness'
            },
            'stress_alert': {
                'frequency_range': (50, 200),
                'consciousness_level': 0.8,
                'meaning': 'High awareness - threat detected'
            },
            'communication_pulse': {
                'frequency_range': (5, 50),
                'consciousness_level': 0.6,
                'meaning': 'Inter-plant communication'
            },
            'photosynthetic_harmony': {
                'frequency_range': (0.5, 5),
                'consciousness_level': 0.4,
                'meaning': 'Light-synchronized consciousness'
            }
        }
    
    def decode_electromagnetic_signals(self, plant_signals: Dict) -> Dict[str, Any]:
        """Main interface for decoding plant electromagnetic signals"""
        if not plant_signals:
            return {'decoded': False, 'message': 'NO_SIGNALS'}
        
        try:
            frequency = plant_signals.get('frequency', 0)
            amplitude = plant_signals.get('amplitude', 0)
            pattern = plant_signals.get('pattern', 'UNKNOWN')
            
            # Detect consciousness pattern
            pattern_type, consciousness_level = self._detect_pattern(frequency)
            
            # Decode message
            decoded_message = self.language_decoder.decode_signal(
                frequency, amplitude, pattern_type
            )
            
            # Create signal record
            signal = PlantSignal(
                timestamp=datetime.now(),
                frequency=frequency,
                amplitude=amplitude,
                pattern_type=pattern_type,
                decoded_message=decoded_message
            )
            
            self.signal_history.append(signal)
            if len(self.signal_history) > 100:
                self.signal_history.pop(0)
            
            return {
                'decoded': True,
                'message': decoded_message,
                'consciousness_level': consciousness_level,
                'pattern_recognized': pattern_type,
                'signal_strength': amplitude
            }
            
        except Exception as e:
            logger.error(f"Signal decode error: {e}")
            return {'decoded': False, 'message': f'DECODE_ERROR: {str(e)}'}
    
    def _detect_pattern(self, frequency: float) -> Tuple[str, float]:
        """Detect consciousness pattern from frequency"""
        best_match = 'unknown'
        highest_consciousness = 0
        
        for pattern_name, pattern_data in self.consciousness_patterns.items():
            freq_range = pattern_data['frequency_range']
            if freq_range[0] <= frequency <= freq_range[1]:
                consciousness_level = pattern_data['consciousness_level']
                if consciousness_level > highest_consciousness:
                    highest_consciousness = consciousness_level
                    best_match = pattern_name
        
        return best_match, highest_consciousness
    
    def monitor_plant_network(self) -> Dict[str, Any]:
        """Monitor plant network status"""
        if not self.signal_history:
            return {'network_active': False, 'coherence': 0}
        
        recent_signals = self.signal_history[-10:]
        consciousness_levels = [
            self.consciousness_patterns.get(s.pattern_type, {}).get('consciousness_level', 0)
            for s in recent_signals
        ]
        
        avg_consciousness = np.mean(consciousness_levels) if consciousness_levels else 0
        
        return {
            'network_active': True,
            'coherence': avg_consciousness,
            'health': 'HEALTHY' if avg_consciousness > 0.3 else 'STRESSED',
            'recent_patterns': [s.pattern_type for s in recent_signals]
        }
    
    def assess_consciousness_level(self) -> float:
        """Assess current plant consciousness level"""
        if not self.signal_history:
            return 0
        
        recent_signals = self.signal_history[-5:]
        consciousness_levels = [
            self.consciousness_patterns.get(s.pattern_type, {}).get('consciousness_level', 0)
            for s in recent_signals
        ]
        
        return np.mean(consciousness_levels) if consciousness_levels else 0

class PlantLanguageDecoder:
    """Decode plant signals into human-readable messages"""
    
    def __init__(self) -> None:
        self.message_templates: Dict[str, List[str]] = {
            'growth_rhythm': [
                "Growing steadily in harmony",
                "Metabolic processes synchronized",
                "Cellular expansion active"
            ],
            'stress_alert': [
                "Environmental threat detected!",
                "Water stress - assistance needed",
                "Temperature stress detected"
            ],
            'communication_pulse': [
                "Inter-plant communication active",
                "Sharing resource data",
                "Coordinating growth patterns"
            ],
            'photosynthetic_harmony': [
                "Synchronized with light cycles",
                "Optimal photosynthesis achieved",
                "Energy production harmonized"
            ],
            'unknown': [
                "Unknown plant communication"
            ]
        }
    
    def decode_signal(self, frequency: float, amplitude: float, pattern_type: str) -> str:
        """Decode plant signal into readable message"""
        templates = self.message_templates.get(pattern_type, self.message_templates['unknown'])
        template_idx = int(amplitude * len(templates)) % len(templates)
        base_message = templates[template_idx]
        
        # Add intensity indicator
        if amplitude > 0.8:
            prefix = "🚨 URGENT: "
        elif amplitude > 0.5:
            prefix = "⚠️ "
        else:
            prefix = "🌿 "
        
        return f"{prefix}{base_message} [F:{frequency:.1f}Hz, A:{amplitude:.2f}]"

if __name__ == "__main__":
    async def demo_plant_communication() -> None:
        plant_interface = PlantCommunicationInterface()
        
        # Simulate plant signals
        test_signals = [
            {'frequency': 1.0, 'amplitude': 0.3, 'pattern': 'GROWTH'},
            {'frequency': 75.0, 'amplitude': 0.9, 'pattern': 'STRESS'},
            {'frequency': 25.0, 'amplitude': 0.6, 'pattern': 'COMMUNICATION'}
        ]
        
        print("🌿 Plant Communication Demo")
        for i, signals in enumerate(test_signals):
            result = plant_interface.decode_electromagnetic_signals(signals)
            print(f"Signal {i+1}: {result['message']}")
        
        # Show network status
        network_status = plant_interface.monitor_plant_network()
        print(f"Network Status: {network_status}")
    
    asyncio.run(demo_plant_communication())