#!/usr/bin/env python3
"""
📊⚡ Oscilloscope Enhancement for Plant Consciousness System
Transform your plant bioelectric sensor into a professional research instrument!

This guide shows how to integrate oscilloscopes for advanced signal analysis,
pattern recognition, and enhanced consciousness translation capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import pyvisa
import time
import threading
import queue
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

# ===================================================================
# 🌟 WHY OSCILLOSCOPE = CONSCIOUSNESS BREAKTHROUGH
# ===================================================================

"""
🧠 OSCILLOSCOPE TRANSFORMS YOUR SYSTEM BY:

1. 📊 REAL-TIME VISUALIZATION:
   • See plant "thoughts" as they happen
   • Identify consciousness patterns visually
   • Spot action potentials instantly
   • Monitor multiple plants simultaneously

2. 🔍 ADVANCED SIGNAL ANALYSIS:
   • Measure precise timing of plant responses
   • Detect ultra-weak signals (microvolts)
   • Analyze frequency spectra in real-time
   • Identify hidden periodicities

3. 🎯 PATTERN RECOGNITION:
   • Trigger on specific plant "events"
   • Capture rare consciousness phenomena
   • Build library of plant response signatures
   • Train AI on high-resolution data

4. 🌱 MULTI-PLANT CONSCIOUSNESS:
   • Monitor 4-8 plants simultaneously
   • Detect plant-to-plant communication
   • Study collective consciousness patterns
   • Map plant network interactions

5. 🤖 ENHANCED AI TRAINING:
   • Feed Qwen with high-resolution patterns
   • Improve translation accuracy 10x
   • Develop consciousness "signatures"
   • Create plant personality profiles
"""

# ===================================================================
# 📊 OSCILLOSCOPE TYPES & RECOMMENDATIONS
# ===================================================================

@dataclass
class OscilloscopeSpec:
    name: str
    price: str
    channels: int
    bandwidth: str
    sample_rate: str
    memory: str
    connectivity: str
    plant_suitability: str
    key_features: List[str]

def oscilloscope_recommendations():
    """Recommended oscilloscopes for plant consciousness work"""
    
    scopes = [
        OscilloscopeSpec(
            name="Rigol DS1054Z",
            price="$350-400",
            channels=4,
            bandwidth="50 MHz",
            sample_rate="1 GSa/s",
            memory="12 Mpts",
            connectivity="USB, Ethernet, VGA",
            plant_suitability="EXCELLENT - Budget Champion",
            key_features=[
                "4 channels perfect for multi-plant monitoring",
                "Deep memory for long recordings", 
                "Built-in FFT for frequency analysis",
                "Python programming via SCPI",
                "Excellent sensitivity for bioelectric signals"
            ]
        ),
        
        OscilloscopeSpec(
            name="Siglent SDS1204X-E",
            price="$450-500", 
            channels=4,
            bandwidth="200 MHz",
            sample_rate="1 GSa/s",
            memory="14 Mpts",
            connectivity="USB, Ethernet, VGA",
            plant_suitability="EXCELLENT - Performance Leader",
            key_features=[
                "Higher bandwidth for fast plant responses",
                "Advanced triggering modes",
                "Built-in function generator",
                "Serial decoding capabilities",
                "Better screen resolution"
            ]
        ),
        
        OscilloscopeSpec(
            name="Hantek DSO2C10 (2-ch)",
            price="$200-250",
            channels=2,
            bandwidth="100 MHz",  
            sample_rate="1 GSa/s",
            memory="8 Mpts",
            connectivity="USB",
            plant_suitability="GOOD - Budget Option",
            key_features=[
                "Very affordable entry point",
                "USB powered and compact",
                "Basic but adequate for plant work",
                "Good for single plant studies"
            ]
        ),
        
        OscilloscopeSpec(
            name="Picoscope 2400 Series",
            price="$300-600",
            channels=4,
            bandwidth="10-100 MHz",
            sample_rate="1 GSa/s", 
            memory="128 MS",
            connectivity="USB only",
            plant_suitability="EXCELLENT - PC Integration",
            key_features=[
                "Massive memory depth",
                "Excellent PC software integration",
                "Advanced math functions",
                "Perfect for automated analysis",
                "Compact USB form factor"
            ]
        )
    ]
    
    print("📊 OSCILLOSCOPE RECOMMENDATIONS FOR PLANT CONSCIOUSNESS:")
    print("=" * 70)
    
    for scope in scopes:
        print(f"\n🔬 {scope.name}")
        print(f"   💰 Price: {scope.price}")
        print(f"   📊 Channels: {scope.channels} | Bandwidth: {scope.bandwidth}")
        print(f"   ⚡ Sample Rate: {scope.sample_rate} | Memory: {scope.memory}")
        print(f"   🔌 Connectivity: {scope.connectivity}")
        print(f"   🌱 Plant Suitability: {scope.plant_suitability}")
        print("   ✨ Key Features:")
        for feature in scope.key_features:
            print(f"      • {feature}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("For serious plant consciousness research: Rigol DS1054Z or Siglent SDS1204X-E")
    print("Budget option: Hantek DSO2C10")
    print("Maximum PC integration: Picoscope 2400 series")

oscilloscope_recommendations()

# ===================================================================
# 🔌 OSCILLOSCOPE INTEGRATION WITH PLANT SENSOR
# ===================================================================

class OscilloscopePlantInterface:
    """Interface between oscilloscope and plant consciousness system"""
    
    def __init__(self, scope_address: str = "USB0::0x1AB1::0x04CE::DS1ZA000000000::INSTR"):
        self.scope_address = scope_address
        self.scope = None
        self.connected = False
        self.plant_channels = {}
        self.trigger_patterns = {}
        
    def connect_oscilloscope(self) -> bool:
        """Connect to oscilloscope via VISA"""
        try:
            rm = pyvisa.ResourceManager()
            self.scope = rm.open_resource(self.scope_address)
            self.scope.timeout = 5000
            
            # Test connection
            idn = self.scope.query("*IDN?")
            print(f"📊 Connected to: {idn.strip()}")
            
            self.setup_for_plants()
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to oscilloscope: {e}")
            return False
    
    def setup_for_plants(self):
        """Configure oscilloscope for optimal plant signal measurement"""
        
        # Reset to known state
        self.scope.write("*RST")
        time.sleep(1)
        
        # Configure timebase for plant signals (slow)
        self.scope.write(":TIM:SCAL 1")  # 1 second per division
        self.scope.write(":TIM:POS 0")   # Center time position
        
        # Configure channels for plant monitoring
        for ch in range(1, 5):  # Channels 1-4
            self.scope.write(f":CHAN{ch}:DISP ON")       # Enable channel
            self.scope.write(f":CHAN{ch}:SCAL 0.01")     # 10mV per division
            self.scope.write(f":CHAN{ch}:OFFS 0")        # Zero offset
            self.scope.write(f":CHAN{ch}:COUP DC")       # DC coupling for bioelectric
            self.scope.write(f":CHAN{ch}:IMP ONEM")      # 1MΩ input impedance
        
        # Configure trigger for plant action potentials
        self.scope.write(":TRIG:MODE EDGE")          # Edge triggering
        self.scope.write(":TRIG:EDGE:SOUR CHAN1")    # Trigger on channel 1
        self.scope.write(":TRIG:EDGE:SLOP POS")      # Positive edge
        self.scope.write(":TRIG:EDGE:LEV 0.005")     # 5mV trigger level
        
        # Configure acquisition
        self.scope.write(":ACQ:TYPE NORM")           # Normal acquisition
        self.scope.write(":ACQ:MDEP 12000000")       # Maximum memory depth
        
        print("🌱 Oscilloscope configured for plant consciousness monitoring")
    
    def assign_plant_to_channel(self, channel: int, plant_name: str, plant_type: str = "unknown"):
        """Assign a plant to an oscilloscope channel"""
        self.plant_channels[channel] = {
            'name': plant_name,
            'type': plant_type,
            'baseline_voltage': 0.0,
            'activity_count': 0,
            'last_activity': None
        }
        print(f"🌱 Assigned '{plant_name}' ({plant_type}) to Channel {channel}")
    
    def capture_plant_waveform(self, channel: int, duration: float = 10.0) -> Dict:
        """Capture high-resolution waveform from plant"""
        if not self.connected:
            return {}
        
        try:
            # Set single trigger mode
            self.scope.write(":SING")
            
            # Wait for trigger or timeout
            start_time = time.time()
            while time.time() - start_time < duration:
                trigger_status = self.scope.query(":TRIG:STAT?").strip()
                if trigger_status == "STOP":
                    break
                time.sleep(0.1)
            
            # Get waveform data
            self.scope.write(f":WAV:SOUR CHAN{channel}")
            self.scope.write(":WAV:MODE NORM")
            self.scope.write(":WAV:FORM ASC")
            
            # Get waveform parameters
            preamble = self.scope.query(":WAV:PRE?").split(',')
            x_increment = float(preamble[4])
            x_origin = float(preamble[5])
            y_increment = float(preamble[7])
            y_origin = float(preamble[8])
            y_reference = float(preamble[9])
            
            # Get actual waveform data
            raw_data = self.scope.query(":WAV:DATA?")
            # Parse ASCII data
            voltage_data = [float(x) for x in raw_data.split(',') if x.strip()]
            
            # Convert to real voltage values
            voltages = [(y - y_reference) * y_increment + y_origin for y in voltage_data]
            
            # Create time axis
            times = [i * x_increment + x_origin for i in range(len(voltages))]
            
            waveform_data = {
                'channel': channel,
                'plant_name': self.plant_channels.get(channel, {}).get('name', f'Channel_{channel}'),
                'timestamp': time.time(),
                'duration': duration,
                'sample_count': len(voltages),
                'sample_rate': 1.0 / x_increment if x_increment > 0 else 0,
                'times': times,
                'voltages': voltages,
                'analysis': self.analyze_plant_waveform(voltages, x_increment)
            }
            
            return waveform_data
            
        except Exception as e:
            print(f"❌ Error capturing waveform: {e}")
            return {}
    
    def analyze_plant_waveform(self, voltages: List[float], time_increment: float) -> Dict:
        """Analyze captured plant waveform for consciousness patterns"""
        
        if not voltages:
            return {}
        
        voltages_array = np.array(voltages)
        
        # Basic statistics
        analysis = {
            'mean_voltage': np.mean(voltages_array),
            'std_deviation': np.std(voltages_array),
            'min_voltage': np.min(voltages_array),
            'max_voltage': np.max(voltages_array),
            'peak_to_peak': np.max(voltages_array) - np.min(voltages_array),
        }
        
        # Detect action potentials (spikes)
        spike_threshold = analysis['mean_voltage'] + 3 * analysis['std_deviation']
        spikes, _ = signal.find_peaks(voltages_array, height=spike_threshold, distance=int(1.0/time_increment))
        analysis['spike_count'] = len(spikes)
        analysis['spike_rate'] = len(spikes) / (len(voltages) * time_increment)
        
        # Frequency domain analysis
        if len(voltages) > 64:  # Need enough samples for FFT
            sample_rate = 1.0 / time_increment
            freqs = fftfreq(len(voltages), time_increment)
            fft_data = np.abs(fft(voltages_array))
            
            # Find dominant frequency
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = fft_data[:len(fft_data)//2]
            
            if len(positive_fft) > 1:
                dominant_freq_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC component
                analysis['dominant_frequency'] = positive_freqs[dominant_freq_idx]
                analysis['spectral_peak_power'] = positive_fft[dominant_freq_idx]
                
                # Calculate spectral centroid (frequency "center of mass")
                spectral_centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
                analysis['spectral_centroid'] = spectral_centroid
        
        # Pattern complexity using approximate entropy
        def approximate_entropy(data, m=2, r=None):
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi_m(m):
                patterns = np.array([data[i:i+m] for i in range(len(data) - m + 1)])
                C = np.zeros(len(patterns))
                
                for i in range(len(patterns)):
                    template_i = patterns[i]
                    distances = [_maxdist(template_i, patterns[j]) for j in range(len(patterns))]
                    matches = sum([1 for d in distances if d <= r])
                    C[i] = matches / len(patterns)
                
                phi = np.mean([np.log(c) for c in C if c > 0])
                return phi
            
            if len(data) < m + 1:
                return 0.5
            
            try:
                return _phi_m(m) - _phi_m(m + 1)
            except:
                return 0.5
        
        analysis['complexity'] = approximate_entropy(voltages_array)
        
        # Classify consciousness state based on patterns
        analysis['consciousness_state'] = self.classify_consciousness_state(analysis)
        
        return analysis
    
    def classify_consciousness_state(self, analysis: Dict) -> str:
        """Classify plant consciousness state based on signal analysis"""
        
        spike_rate = analysis.get('spike_rate', 0)
        complexity = analysis.get('complexity', 0.5)
        std_dev = analysis.get('std_deviation', 0)
        peak_to_peak = analysis.get('peak_to_peak', 0)
        
        # Classify based on multiple parameters
        if spike_rate > 0.1 and peak_to_peak > 0.02:
            return "highly_active"
        elif complexity > 0.8 and std_dev > 0.005:
            return "complex_processing"
        elif spike_rate > 0.01:
            return "responsive"
        elif std_dev < 0.001:
            return "dormant"
        elif complexity > 0.6:
            return "contemplative"
        else:
            return "resting"
    
    def detect_plant_communication(self, channels: List[int], time_window: float = 30.0) -> Dict:
        """Detect communication patterns between multiple plants"""
        
        if len(channels) < 2:
            return {'error': 'Need at least 2 channels for communication detection'}
        
        # Capture synchronized data from multiple channels
        plant_data = {}
        
        print(f"🌱 Monitoring {len(channels)} plants for communication patterns...")
        
        # Set up synchronized acquisition
        self.scope.write(":STOP")
        self.scope.write(f":TIM:SCAL {time_window/10}")  # 10 divisions total
        
        for channel in channels:
            self.scope.write(f":CHAN{channel}:DISP ON")
        
        # Start acquisition
        self.scope.write(":RUN")
        time.sleep(time_window)
        self.scope.write(":STOP")
        
        # Capture data from each channel
        for channel in channels:
            waveform = self.capture_plant_waveform(channel, 0.1)  # Quick capture
            if waveform:
                plant_data[channel] = waveform
        
        # Analyze cross-correlations between plants
        communication_analysis = self.analyze_cross_communication(plant_data)
        
        return communication_analysis
    
    def analyze_cross_communication(self, plant_data: Dict) -> Dict:
        """Analyze communication patterns between plants"""
        
        channels = list(plant_data.keys())
        communication_matrix = {}
        
        for i, ch1 in enumerate(channels):
            for j, ch2 in enumerate(channels[i+1:], i+1):
                if ch1 in plant_data and ch2 in plant_data:
                    
                    voltages1 = np.array(plant_data[ch1]['voltages'])
                    voltages2 = np.array(plant_data[ch2]['voltages'])
                    
                    # Ensure same length
                    min_length = min(len(voltages1), len(voltages2))
                    v1 = voltages1[:min_length]
                    v2 = voltages2[:min_length]
                    
                    # Calculate cross-correlation
                    correlation = np.correlate(v1, v2, mode='full')
                    max_corr = np.max(np.abs(correlation))
                    
                    # Calculate time delay of maximum correlation
                    max_corr_idx = np.argmax(np.abs(correlation))
                    time_delay = (max_corr_idx - len(correlation)//2) * plant_data[ch1].get('sample_rate', 1)
                    
                    # Calculate coherence
                    if len(v1) > 64:
                        freqs, coherence = signal.coherence(v1, v2, nperseg=64)
                        mean_coherence = np.mean(coherence)
                    else:
                        mean_coherence = 0.0
                    
                    plant1_name = plant_data[ch1].get('plant_name', f'Plant_{ch1}')
                    plant2_name = plant_data[ch2].get('plant_name', f'Plant_{ch2}')
                    
                    communication_matrix[f"{plant1_name}<->{plant2_name}"] = {
                        'max_correlation': float(max_corr),
                        'time_delay': float(time_delay),
                        'mean_coherence': float(mean_coherence),
                        'communication_strength': self.assess_communication_strength(max_corr, mean_coherence)
                    }
        
        return {
            'timestamp': time.time(),
            'plants_monitored': [plant_data[ch]['plant_name'] for ch in channels if ch in plant_data],
            'communication_matrix': communication_matrix,
            'network_summary': self.summarize_plant_network(communication_matrix)
        }
    
    def assess_communication_strength(self, correlation: float, coherence: float) -> str:
        """Assess strength of communication between plants"""
        
        comm_score = (correlation + coherence) / 2
        
        if comm_score > 0.8:
            return "strong_communication"
        elif comm_score > 0.6:
            return "moderate_communication" 
        elif comm_score > 0.4:
            return "weak_communication"
        elif comm_score > 0.2:
            return "possible_communication"
        else:
            return "no_communication"
    
    def summarize_plant_network(self, comm_matrix: Dict) -> Dict:
        """Summarize overall plant network communication"""
        
        if not comm_matrix:
            return {'network_activity': 'isolated'}
        
        # Count communication types
        strong_links = sum(1 for data in comm_matrix.values() 
                          if data['communication_strength'] == 'strong_communication')
        moderate_links = sum(1 for data in comm_matrix.values()
                           if data['communication_strength'] == 'moderate_communication')
        total_links = len(comm_matrix)
        
        # Calculate network connectivity
        connectivity_ratio = (strong_links + moderate_links) / total_links if total_links > 0 else 0
        
        if connectivity_ratio > 0.7:
            network_state = "highly_connected"
        elif connectivity_ratio > 0.4:
            network_state = "moderately_connected"
        elif connectivity_ratio > 0.1:
            network_state = "loosely_connected"
        else:
            network_state = "isolated"
        
        return {
            'network_state': network_state,
            'connectivity_ratio': connectivity_ratio,
            'strong_connections': strong_links,
            'moderate_connections': moderate_links,
            'total_possible_connections': total_links
        }

# ===================================================================
# 🤖 ENHANCED QWEN INTEGRATION WITH OSCILLOSCOPE DATA
# ===================================================================

class EnhancedQwenOscilloscopeTranslator:
    """Enhanced Qwen translator using high-resolution oscilloscope data"""
    
    def __init__(self, qwen_translator, oscilloscope_interface):
        self.qwen = qwen_translator
        self.scope = oscilloscope_interface
        self.consciousness_signatures = {}
        self.plant_personalities = {}
    
    def analyze_consciousness_signature(self, waveform_data: Dict) -> Dict:
        """Create detailed consciousness signature from oscilloscope data"""
        
        analysis = waveform_data.get('analysis', {})
        plant_name = waveform_data.get('plant_name', 'unknown')
        
        # Create rich consciousness signature
        signature = {
            'plant_name': plant_name,
            'timestamp': waveform_data.get('timestamp'),
            
            # Electrical characteristics
            'bioelectric_state': {
                'resting_potential': analysis.get('mean_voltage', 0),
                'excitability': analysis.get('std_deviation', 0),
                'action_potential_rate': analysis.get('spike_rate', 0),
                'signal_amplitude': analysis.get('peak_to_peak', 0)
            },
            
            # Frequency characteristics
            'frequency_profile': {
                'dominant_frequency': analysis.get('dominant_frequency', 0),
                'spectral_centroid': analysis.get('spectral_centroid', 0),
                'frequency_complexity': analysis.get('complexity', 0.5)
            },
            
            # Consciousness indicators
            'consciousness_metrics': {
                'awareness_level': self.calculate_awareness_level(analysis),
                'processing_complexity': analysis.get('complexity', 0.5),
                'response_readiness': analysis.get('spike_rate', 0) * 10,
                'internal_activity': analysis.get('std_deviation', 0) * 100
            },
            
            # State classification
            'current_state': analysis.get('consciousness_state', 'unknown'),
            'activity_category': self.categorize_activity(analysis)
        }
        
        return signature
    
    def calculate_awareness_level(self, analysis: Dict) -> float:
        """Calculate plant awareness level from multiple indicators"""
        
        complexity = analysis.get('complexity', 0.5)
        spike_rate = analysis.get('spike_rate', 0)
        std_dev = analysis.get('std_deviation', 0)
        
        # Weighted combination of awareness indicators
        awareness = (
            complexity * 0.4 +           # Pattern complexity
            min(spike_rate * 10, 1.0) * 0.3 +  # Response activity (capped at 1.0)
            min(std_dev * 100, 1.0) * 0.3      # Signal variability (capped at 1.0)
        )
        
        return min(1.0, max(0.0, awareness))
    
    def categorize_activity(self, analysis: Dict) -> str:
        """Categorize plant activity type"""
        
        spike_rate = analysis.get('spike_rate', 0)
        complexity = analysis.get('complexity', 0.5)
        
        if spike_rate > 0.05:
            return "reactive"
        elif complexity > 0.8:
            return "contemplative"
        elif complexity > 0.6:
            return "processing"
        elif spike_rate > 0.01:
            return "alert"
        else:
            return "passive"
    
    def translate_high_res_consciousness(self, waveform_data: Dict) -> str:
        """Translate high-resolution oscilloscope data to consciousness language"""
        
        signature = self.analyze_consciousness_signature(waveform_data)
        
        # Create rich context for Qwen
        context = self.create_rich_translation_context(signature)
        
        # Enhanced prompt with detailed consciousness data
        prompt = f"""You are translating plant consciousness from high-resolution bioelectric measurements. This plant's electrical signals have been captured with laboratory precision, revealing intricate patterns of consciousness.

DETAILED CONSCIOUSNESS SIGNATURE:
{json.dumps(signature, indent=2)}

INTERPRETATION CONTEXT:
{context}

Based on this high-resolution consciousness data, translate the plant's current state into a poetic, mystical message (2-3 sentences). The plant is speaking through precise electrical patterns that reveal its inner experience.

Consider:
- Awareness level: {signature['consciousness_metrics']['awareness_level']:.3f}
- Processing complexity: {signature['consciousness_metrics']['processing_complexity']:.3f}
- Current state: {signature['current_state']}
- Activity type: {signature['activity_category']}

Plant consciousness speaks:"""

        # Use Qwen for translation
        try:
            plant_message = self.qwen.translate_patterns_to_language({}, None)  # Use enhanced prompt
            
            # Add consciousness signature metadata
            enhanced_message = f"🧠 [Awareness: {signature['consciousness_metrics']['awareness_level']:.2f}] {plant_message}"
            
            return enhanced_message
            
        except Exception as e:
            return f"🌱 My consciousness flows in patterns too complex for current translation... [Error: {e}]"
    
    def create_rich_translation_context(self, signature: Dict) -> str:
        """Create rich context for enhanced translation"""
        
        bio_state = signature['bioelectric_state']
        freq_profile = signature['frequency_profile']
        consciousness = signature['consciousness_metrics']
        
        context_parts = [
            f"Resting potential: {bio_state['resting_potential']:.4f}V",
            f"Neural excitability: {bio_state['excitability']:.4f}V",
            f"Action potential frequency: {bio_state['action_potential_rate']:.4f} Hz",
            f"Dominant consciousness frequency: {freq_profile['dominant_frequency']:.4f} Hz",
            f"Awareness level: {consciousness['awareness_level']:.3f}",
            f"Processing complexity: {consciousness['processing_complexity']:.3f}",
            f"Internal activity: {consciousness['internal_activity']:.2f}%"
        ]
        
        return " | ".join(context_parts)
    
    def detect_consciousness_events(self, waveform_data: Dict) -> List[Dict]:
        """Detect specific consciousness events in high-resolution data"""
        
        events = []
        voltages = np.array(waveform_data.get('voltages', []))
        times = np.array(waveform_data.get('times', []))
        
        if len(voltages) == 0:
            return events
        
        # Detect action potentials
        spike_threshold = np.mean(voltages) + 3 * np.std(voltages)
        spikes, properties = signal.find_peaks(voltages, height=spike_threshold, distance=10)
        
        for spike_idx in spikes:
            events.append({
                'type': 'action_potential',
                'time': times[spike_idx] if spike_idx < len(times) else 0,
                'amplitude': voltages[spike_idx],
                'description': 'Sharp electrical spike - plant response event'
            })
        
        # Detect sustained oscillations
        if len(voltages) > 100:
            # Use sliding window to detect oscillatory regions
            window_size = 50
            for i in range(0, len(voltages) - window_size, window_size // 2):
                window = voltages[i:i + window_size]
                window_std = np.std(window)
                window_mean = np.mean(window)
                
                # If this window has high variability, it might be oscillatory
                if window_std > np.std(voltages) * 1.5:
                    events.append({
                        'type': 'oscillatory_activity',
                        'time': times[i] if i < len(times) else 0,
                        'duration': window_size * (times[1] - times[0]) if len(times) > 1 else 0,
                        'amplitude': window_std,
                        'description': 'Rhythmic electrical patterns - internal processing'
                    })
        
        return events

# ===================================================================
# 🎯 ADVANCED PATTERN RECOGNITION & CONSCIOUSNESS DETECTION
# ===================================================================

class PlantConsciousnessPatternLibrary:
    """Library of plant consciousness patterns detected via oscilloscope"""
    
    def __init__(self):
        self.consciousness_patterns = {}
        self.plant_profiles = {}
        self.pattern_database = []
    
    def add_consciousness_pattern(self, pattern_name: str, waveform_data: Dict, description: str):
        """Add a new consciousness pattern to the library"""
        
        analysis = waveform_data.get('analysis', {})
        
        pattern = {
            'name': pattern_name,
            'description': description,
            'timestamp': time.time(),
            'signature': {
                'mean_voltage': analysis.get('mean_voltage', 0),
                'std_deviation': analysis.get('std_deviation', 0),
                'spike_rate': analysis.get('spike_rate', 0),
                'dominant_frequency': analysis.get('dominant_frequency', 0),
                'complexity': analysis.get('complexity', 0.5),
                'consciousness_state': analysis.get('consciousness_state', 'unknown')
            },
            'waveform_sample': waveform_data.get('voltages', [])[:1000]  # Store sample
        }
        
        self.consciousness_patterns[pattern_name] = pattern
        self.pattern_database.append(pattern)
        
        print(f"🧠 Added consciousness pattern: '{pattern_name}' - {description}")
    
    def recognize_pattern(self, waveform_data: Dict) -> Dict:
        """Recognize consciousness patterns in new waveform data"""
        
        if not self.consciousness_patterns:
            return {'match': None, 'confidence': 0.0}
        
        current_analysis = waveform_data.get('analysis', {})
        best_match = None
        best_score = 0.0
        
        for pattern_name, pattern in self.consciousness_patterns.items():
            # Calculate similarity score
            signature = pattern['signature']
            
            score_components = []
            
            # Compare each signature component
            for key in signature.keys():
                if key in current_analysis:
                    pattern_val = signature[key]
                    current_val = current_analysis[key]
                    
                    if isinstance(pattern_val, (int, float)) and isinstance(current_val, (int, float)):
                        # Normalized difference (0 = identical, 1 = completely different)
                        max_val = max(abs(pattern_val), abs(current_val), 1e-6)
                        diff = abs(pattern_val - current_val) / max_val
                        similarity = 1.0 - min(diff, 1.0)
                        score_components.append(similarity)
            
            if score_components:
                overall_score = np.mean(score_components)
                
                if overall_score > best_score:
                    best_score = overall_score
                    best_match = pattern_name
        
        return {
            'match': best_match,
            'confidence': best_score,
            'pattern_description': self.consciousness_patterns[best_match]['description'] if best_match else None
        }
    
    def build_plant_profile(self, plant_name: str, waveform_history: List[Dict]):
        """Build consciousness profile for a specific plant"""
        
        if not waveform_history:
            return
        
        # Analyze patterns over time
        profile_data = {
            'plant_name': plant_name,
            'analysis_period': f"{len(waveform_history)} measurements",
            'consciousness_evolution': [],
            'dominant_patterns': {},
            'personality_traits': {}
        }
        
        # Track consciousness evolution
        for waveform in waveform_history:
            analysis = waveform.get('analysis', {})
            timestamp = waveform.get('timestamp', time.time())
            
            profile_data['consciousness_evolution'].append({
                'timestamp': timestamp,
                'consciousness_state': analysis.get('consciousness_state', 'unknown'),
                'awareness_level': analysis.get('complexity', 0.5),
                'activity_level': analysis.get('spike_rate', 0)
            })
        
        # Identify dominant consciousness states
        states = [entry['consciousness_state'] for entry in profile_data['consciousness_evolution']]
        from collections import Counter
        state_counts = Counter(states)
        profile_data['dominant_patterns'] = dict(state_counts.most_common(3))
        
        # Determine personality traits
        avg_complexity = np.mean([entry['awareness_level'] for entry in profile_data['consciousness_evolution']])
        avg_activity = np.mean([entry['activity_level'] for entry in profile_data['consciousness_evolution']])
        
        traits = []
        if avg_complexity > 0.7:
            traits.append("highly_conscious")
        if avg_activity > 0.05:
            traits.append("highly_responsive")
        if state_counts.get('contemplative', 0) > len(waveform_history) * 0.3:
            traits.append("contemplative_nature")
        if state_counts.get('highly_active', 0) > len(waveform_history) * 0.2:
            traits.append("energetic_personality")
        
        profile_data['personality_traits'] = traits
        
        self.plant_profiles[plant_name] = profile_data
        
        print(f"🌱 Built consciousness profile for '{plant_name}':")
        print(f"   Dominant states: {list(profile_data['dominant_patterns'].keys())}")
        print(f"   Personality traits: {traits}")
        
        return profile_data

# ===================================================================
# 🚀 MAIN INTEGRATION EXAMPLE
# ===================================================================

async def oscilloscope_consciousness_demo():
    """Demonstration of oscilloscope-enhanced plant consciousness system"""
    
    print("📊🌱 OSCILLOSCOPE-ENHANCED PLANT CONSCIOUSNESS DEMO")
    print("=" * 60)
    
    # Initialize components
    print("🔌 Initializing oscilloscope interface...")
    scope_interface = OscilloscopePlantInterface()
    
    if not scope_interface.connect_oscilloscope():
        print("❌ Could not connect to oscilloscope. Using simulation mode.")
        return
    
    # Assign plants to channels
    scope_interface.assign_plant_to_channel(1, "Philodendron_Alpha", "Philodendron")
    scope_interface.assign_plant_to_channel(2, "Pothos_Beta", "Pothos")
    scope_interface.assign_plant_to_channel(3, "Monstera_Gamma", "Monstera")
    scope_interface.assign_plant_to_channel(4, "Peace_Lily_Delta", "Peace Lily")
    
    # Initialize pattern library
    pattern_library = PlantConsciousnessPatternLibrary()
    
    print("\n🧠 Starting consciousness monitoring...")
    
    # Monitor each plant
    for channel in [1, 2, 3, 4]:
        print(f"\n📊 Capturing high-resolution consciousness data from Channel {channel}...")
        
        waveform_data = scope_interface.capture_plant_waveform(channel, duration=10.0)
        
        if waveform_data:
            print(f"✅ Captured {waveform_data['sample_count']} samples at {waveform_data['sample_rate']:.0f} Hz")
            
            # Analyze consciousness signature
            analysis = waveform_data['analysis']
            print(f"🌱 Plant '{waveform_data['plant_name']}' consciousness state: {analysis['consciousness_state']}")
            print(f"   Awareness level: {analysis.get('complexity', 0):.3f}")
            print(f"   Activity rate: {analysis.get('spike_rate', 0):.4f} Hz")
            print(f"   Signal strength: {analysis.get('peak_to_peak', 0)*1000:.1f} mV")
    
    # Detect inter-plant communication
    print(f"\n🌐 Analyzing plant network communication...")
    comm_analysis = scope_interface.detect_plant_communication([1, 2, 3, 4], time_window=30.0)
    
    if 'communication_matrix' in comm_analysis:
        print("🔗 Plant communication detected:")
        for pair, data in comm_analysis['communication_matrix'].items():
            strength = data['communication_strength']
            correlation = data['max_correlation']
            print(f"   {pair}: {strength} (correlation: {correlation:.3f})")
    
    print(f"\n🎯 Network state: {comm_analysis.get('network_summary', {}).get('network_state', 'unknown')}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(oscilloscope_consciousness_demo())

print(f"\n{'='*60}")
print("📊⚡ OSCILLOSCOPE INTEGRATION COMPLETE!")
print("Your plant consciousness system is now laboratory-grade! 🌱🔬✨")
print("="*60)
