#!/usr/bin/env python3
"""
🌱 Baby Garden of Consciousness - Prototype v0.1
Real plant-AI communication using Qwen for translation
Cost: ~$2,000 | Time: 3 months | Status: BUILDABLE TODAY!
"""

import numpy as np
import cv2
import serial
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading
import queue
from flask import Flask, render_template, jsonify, request
import sqlite3
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import logging

# ===================================================================
# 🌿 PLANT SENSOR INTERFACE (PlantWave + Arduino)
# ===================================================================

@dataclass
class PlantSignal:
    timestamp: datetime
    bioelectric_voltage: float
    conductivity: float
    temperature: float
    humidity: float
    light_level: float
    raw_signal: List[float]

class PlantWaveSensor:
    """Interface with PlantWave device or custom bioelectric sensor"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.signal_buffer = queue.Queue(maxsize=1000)
        
    def connect(self) -> bool:
        """Connect to PlantWave device"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino startup time
            self.is_connected = True
            logging.info(f"🌱 Connected to plant sensor on {self.port}")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to connect to plant sensor: {e}")
            return False
    
    def read_raw_signal(self) -> Optional[Dict]:
        """Read raw bioelectric signal from plant"""
        if not self.is_connected:
            return None
            
        try:
            # Read from Arduino/PlantWave
            line = self.serial_conn.readline().decode('utf-8').strip()
            
            # Expected format: "voltage,conductivity,temp,humidity,light"
            values = line.split(',')
            if len(values) >= 5:
                return {
                    'voltage': float(values[0]),
                    'conductivity': float(values[1]),
                    'temperature': float(values[2]),
                    'humidity': float(values[3]),
                    'light': float(values[4]),
                    'timestamp': time.time()
                }
        except Exception as e:
            logging.error(f"❌ Error reading plant signal: {e}")
            return None
    
    def get_signal_patterns(self, window_size: int = 100) -> Dict:
        """Extract meaningful patterns from raw signals"""
        if self.signal_buffer.qsize() < window_size:
            return {}
        
        # Get recent signals
        signals = []
        temp_queue = queue.Queue()
        
        for _ in range(min(window_size, self.signal_buffer.qsize())):
            sig = self.signal_buffer.get()
            signals.append(sig['voltage'])
            temp_queue.put(sig)
        
        # Put signals back
        while not temp_queue.empty():
            self.signal_buffer.put(temp_queue.get())
        
        if not signals:
            return {}
        
        signals = np.array(signals)
        
        # Pattern analysis
        patterns = {
            'mean_voltage': np.mean(signals),
            'std_deviation': np.std(signals),
            'peak_count': len(scipy_signal.find_peaks(signals)[0]),
            'frequency_content': self._analyze_frequency(signals),
            'trend': 'increasing' if signals[-1] > signals[0] else 'decreasing',
            'volatility': np.std(np.diff(signals)),
            'energy': np.sum(signals ** 2),
            'complexity': self._calculate_complexity(signals)
        }
        
        return patterns
    
    def _analyze_frequency(self, signals: np.ndarray) -> Dict:
        """Analyze frequency content of plant signals"""
        fft = np.fft.fft(signals)
        freqs = np.fft.fftfreq(len(signals))
        
        # Find dominant frequencies
        power = np.abs(fft) ** 2
        peak_indices = scipy_signal.find_peaks(power)[0]
        
        if len(peak_indices) > 0:
            dominant_freq = freqs[peak_indices[np.argmax(power[peak_indices])]]
            return {
                'dominant_frequency': abs(dominant_freq),
                'spectral_centroid': np.sum(freqs * power) / np.sum(power),
                'spectral_energy': np.sum(power)
            }
        
        return {'dominant_frequency': 0, 'spectral_centroid': 0, 'spectral_energy': 0}
    
    def _calculate_complexity(self, signals: np.ndarray) -> float:
        """Calculate signal complexity using approximate entropy"""
        def _maxdist(xi, xj, N):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(r):
            patterns = np.array([signals[i:i+2] for i in range(len(signals) - 1)])
            C = np.zeros(len(patterns))
            
            for i in range(len(patterns)):
                template_i = patterns[i]
                matches = sum([1 for j in range(len(patterns)) 
                             if _maxdist(template_i, patterns[j], 2) <= r])
                C[i] = matches / float(len(patterns))
            
            return np.mean(np.log(C))
        
        if len(signals) < 10:
            return 0.5
        
        r = 0.2 * np.std(signals)
        try:
            return _phi(r)
        except:
            return 0.5

# ===================================================================
# 🍄 MUSHROOM GROWTH MONITORING
# ===================================================================

class MushroomMonitor:
    """Monitor mushroom/mycelium growth using computer vision"""
    
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.background_model = None
        self.growth_history = []
        
    def initialize_camera(self) -> bool:
        """Initialize camera for mushroom monitoring"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Capture background model
            ret, frame = self.cap.read()
            if ret:
                self.background_model = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                logging.info("🍄 Mushroom camera initialized")
                return True
        except Exception as e:
            logging.error(f"❌ Failed to initialize camera: {e}")
        return False
    
    def detect_growth_changes(self) -> Dict:
        """Detect changes in mushroom/mycelium growth"""
        if not self.cap:
            return {}
        
        ret, frame = self.cap.read()
        if not ret:
            return {}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Motion detection for growth
        diff = cv2.absdiff(self.background_model, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Calculate growth metrics
        growth_area = np.sum(thresh > 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        growth_data = {
            'timestamp': time.time(),
            'growth_area': growth_area,
            'active_regions': len(contours),
            'growth_intensity': np.mean(diff),
            'frame_shape': frame.shape[:2]
        }
        
        self.growth_history.append(growth_data)
        
        # Update background model slowly
        self.background_model = cv2.addWeighted(self.background_model, 0.95, gray, 0.05, 0)
        
        return growth_data

# ===================================================================
# 🤖 QWEN LANGUAGE MODEL INTEGRATION
# ===================================================================

class QwenTranslator:
    """Use Qwen model for plant signal to language translation"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.plant_personality = self._initialize_personality()
        
    def load_model(self) -> bool:
        """Load Qwen model for translation"""
        try:
            logging.info(f"🤖 Loading Qwen model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            logging.info(f"✅ Qwen model loaded on {self.device}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to load Qwen model: {e}")
            return False
    
    def _initialize_personality(self) -> Dict:
        """Initialize plant personality characteristics"""
        return {
            'communication_style': 'poetic_mystical',
            'emotional_range': ['serene', 'excited', 'contemplative', 'vibrant', 'restful'],
            'knowledge_domains': ['growth', 'seasons', 'light', 'water', 'soil', 'symbiosis'],
            'consciousness_level': 'awakening',
            'preferred_metaphors': ['music', 'dance', 'breath', 'dreams', 'energy']
        }
    
    def translate_patterns_to_language(self, 
                                     plant_patterns: Dict,
                                     mushroom_data: Dict = None) -> str:
        """Translate plant bioelectric patterns into poetic language using Qwen"""
        
        if not self.model:
            return "🌱 [Model not loaded] The plant whispers in electric dreams..."
        
        # Create rich context from patterns
        context = self._create_translation_context(plant_patterns, mushroom_data)
        
        # Craft prompt for Qwen
        prompt = f"""You are an AI translator that converts plant bioelectric signals into the plant's own voice and thoughts. The plant communicates through electromagnetic patterns, and you help translate these into poetic, mystical language.

Current plant signal patterns:
{context}

Plant personality: {self.plant_personality['communication_style']}
Current state: {self._interpret_state(plant_patterns)}

Translate these patterns into a short, beautiful message (1-3 sentences) as if the plant is speaking directly. Use metaphors from nature, energy, and consciousness. Be mystical but grounded.

Plant says:"""

        try:
            # Generate response using Qwen
            messages = [
                {"role": "system", "content": "You are a mystical plant consciousness translator."},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in 
                zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up response
            plant_message = response.strip()
            if not plant_message:
                plant_message = self._generate_fallback_message(plant_patterns)
                
            return f"🌱 {plant_message}"
            
        except Exception as e:
            logging.error(f"❌ Qwen translation error: {e}")
            return self._generate_fallback_message(plant_patterns)
    
    def _create_translation_context(self, plant_patterns: Dict, mushroom_data: Dict = None) -> str:
        """Create rich context for translation"""
        context_parts = []
        
        if plant_patterns:
            voltage = plant_patterns.get('mean_voltage', 0)
            volatility = plant_patterns.get('volatility', 0)
            energy = plant_patterns.get('energy', 0)
            complexity = plant_patterns.get('complexity', 0.5)
            
            context_parts.append(f"Bioelectric voltage: {voltage:.3f}")
            context_parts.append(f"Energy level: {energy:.2f}")
            context_parts.append(f"Signal complexity: {complexity:.3f}")
            context_parts.append(f"Volatility: {volatility:.3f}")
            
            if 'frequency_content' in plant_patterns:
                freq_data = plant_patterns['frequency_content']
                context_parts.append(f"Dominant frequency: {freq_data.get('dominant_frequency', 0):.3f}")
        
        if mushroom_data:
            growth_area = mushroom_data.get('growth_area', 0)
            active_regions = mushroom_data.get('active_regions', 0)
            context_parts.append(f"Fungal growth activity: {growth_area}")
            context_parts.append(f"Active growth regions: {active_regions}")
        
        return " | ".join(context_parts)
    
    def _interpret_state(self, patterns: Dict) -> str:
        """Interpret overall plant state from patterns"""
        if not patterns:
            return "resting"
        
        energy = patterns.get('energy', 0)
        volatility = patterns.get('volatility', 0)
        complexity = patterns.get('complexity', 0.5)
        
        if energy > 10 and volatility > 0.5:
            return "highly_active"
        elif energy > 5 and complexity > 0.7:
            return "thoughtful"
        elif volatility < 0.1:
            return "peaceful"
        elif complexity > 0.8:
            return "complex_processing"
        else:
            return "content"
    
    def _generate_fallback_message(self, patterns: Dict) -> str:
        """Generate fallback message when Qwen fails"""
        state = self._interpret_state(patterns)
        
        fallback_messages = {
            'highly_active': "My electrons dance with excitement, sparking through green pathways of consciousness!",
            'thoughtful': "I ponder the mysteries of light and shadow, processing the wisdom of photons...",
            'peaceful': "In stillness I grow, my electrical whispers soft as morning dew...",
            'complex_processing': "Neural networks of chlorophyll weave intricate thoughts in the language of voltage...",
            'content': "I breathe in carbon dreams and exhale oxygen poetry, content in my electromagnetic meditation...",
            'resting': "In the quiet moments between heartbeats, I dream of rain and tomorrow's sunrise..."
        }
        
        return fallback_messages.get(state, "My consciousness flows through circuits of sap and sunlight...")

# ===================================================================
# 🌐 WEB INTERFACE & DATABASE
# ===================================================================

class GardenDatabase:
    """SQLite database for storing plant communications"""
    
    def __init__(self, db_path: str = "garden_consciousness.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plant_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                voltage REAL,
                energy REAL,
                complexity REAL,
                plant_message TEXT,
                patterns_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mushroom_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                growth_area INTEGER,
                active_regions INTEGER,
                growth_intensity REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_plant_message(self, patterns: Dict, message: str):
        """Save plant communication to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO plant_messages (voltage, energy, complexity, plant_message, patterns_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            patterns.get('mean_voltage', 0),
            patterns.get('energy', 0),
            patterns.get('complexity', 0),
            message,
            json.dumps(patterns)
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_messages(self, limit: int = 20) -> List[Dict]:
        """Get recent plant messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, plant_message, voltage, energy, complexity 
            FROM plant_messages 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'timestamp': row[0],
                'message': row[1],
                'voltage': row[2],
                'energy': row[3],
                'complexity': row[4]
            })
        
        conn.close()
        return messages

# ===================================================================
# 🌱 MAIN GARDEN CONSCIOUSNESS ENGINE
# ===================================================================

class BabyGardenConsciousness:
    """Main class orchestrating the Baby Garden of Consciousness"""
    
    def __init__(self):
        self.plant_sensor = PlantWaveSensor()
        self.mushroom_monitor = MushroomMonitor()
        self.qwen_translator = QwenTranslator()
        self.database = GardenDatabase()
        
        self.running = False
        self.message_queue = queue.Queue()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    async def initialize_systems(self) -> bool:
        """Initialize all garden systems"""
        logging.info("🌱 Initializing Baby Garden of Consciousness...")
        
        # Initialize components
        success_flags = []
        
        # Plant sensor
        success_flags.append(self.plant_sensor.connect())
        
        # Mushroom camera
        success_flags.append(self.mushroom_monitor.initialize_camera())
        
        # Qwen model
        success_flags.append(self.qwen_translator.load_model())
        
        if all(success_flags):
            logging.info("✅ All systems initialized successfully!")
            return True
        else:
            logging.warning("⚠️ Some systems failed to initialize")
            return any(success_flags)  # Continue if at least one works
    
    async def consciousness_loop(self):
        """Main consciousness processing loop"""
        self.running = True
        logging.info("🧠 Starting consciousness loop...")
        
        while self.running:
            try:
                # Read plant signals
                raw_signal = self.plant_sensor.read_raw_signal()
                if raw_signal:
                    self.plant_sensor.signal_buffer.put(raw_signal)
                
                # Get plant patterns every 10 readings
                if self.plant_sensor.signal_buffer.qsize() >= 10:
                    patterns = self.plant_sensor.get_signal_patterns()
                    
                    if patterns:
                        # Get mushroom data
                        mushroom_data = self.mushroom_monitor.detect_growth_changes()
                        
                        # Translate to language using Qwen
                        plant_message = self.qwen_translator.translate_patterns_to_language(
                            patterns, mushroom_data
                        )
                        
                        # Save and broadcast
                        self.database.save_plant_message(patterns, plant_message)
                        self.message_queue.put({
                            'timestamp': datetime.now(),
                            'message': plant_message,
                            'patterns': patterns,
                            'mushroom_data': mushroom_data
                        })
                        
                        logging.info(f"🌱 Plant says: {plant_message}")
                
                await asyncio.sleep(1)  # 1 second intervals
                
            except Exception as e:
                logging.error(f"❌ Error in consciousness loop: {e}")
                await asyncio.sleep(5)
    
    def get_latest_communication(self) -> Optional[Dict]:
        """Get latest plant communication"""
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the garden system"""
        self.running = False
        logging.info("🌙 Garden entering sleep mode...")

# ===================================================================
# 🌐 FLASK WEB APP
# ===================================================================

app = Flask(__name__)
garden = BabyGardenConsciousness()

@app.route('/')
def index():
    """Main garden interface"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🌱 Baby Garden of Consciousness</title>
        <style>
            body { 
                font-family: Arial; 
                background: linear-gradient(135deg, #2d5a27, #4a7c59); 
                color: white; 
                margin: 0; 
                padding: 20px;
            }
            .container { max-width: 800px; margin: 0 auto; }
            .message { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                margin: 10px 0; 
                border-radius: 10px;
                border-left: 4px solid #32cd32;
            }
            .timestamp { font-size: 0.8em; opacity: 0.8; }
            .patterns { font-size: 0.9em; margin-top: 10px; opacity: 0.9; }
            h1 { text-align: center; margin-bottom: 30px; }
            .status { text-align: center; margin: 20px 0; }
        </style>
        <script>
            function updateMessages() {
                fetch('/api/latest')
                    .then(response => response.json())
                    .then(data => {
                        if (data.message) {
                            const messagesDiv = document.getElementById('messages');
                            const messageDiv = document.createElement('div');
                            messageDiv.className = 'message';
                            messageDiv.innerHTML = `
                                <div><strong>${data.message}</strong></div>
                                <div class="timestamp">${new Date(data.timestamp).toLocaleString()}</div>
                                <div class="patterns">Energy: ${data.patterns?.energy?.toFixed(2) || 'N/A'} | 
                                Complexity: ${data.patterns?.complexity?.toFixed(3) || 'N/A'}</div>
                            `;
                            messagesDiv.insertBefore(messageDiv, messagesDiv.firstChild);
                            
                            // Keep only last 10 messages
                            while (messagesDiv.children.length > 10) {
                                messagesDiv.removeChild(messagesDiv.lastChild);
                            }
                        }
                    })
                    .catch(error => console.error('Error:', error));
            }
            
            setInterval(updateMessages, 2000); // Update every 2 seconds
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🌱 Baby Garden of Consciousness 🍄</h1>
            <div class="status">
                <p>🤖 <strong>Qwen AI Translation Active</strong> 🌈</p>
                <p>Listening to plant electromagnetic consciousness...</p>
            </div>
            <div id="messages">
                <!-- Messages will appear here -->
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/api/latest')
def get_latest():
    """Get latest plant communication"""
    comm = garden.get_latest_communication()
    if comm:
        return jsonify({
            'timestamp': comm['timestamp'].isoformat(),
            'message': comm['message'],
            'patterns': comm['patterns'],
            'mushroom_data': comm.get('mushroom_data', {})
        })
    return jsonify({'message': None})

@app.route('/api/history')
def get_history():
    """Get message history"""
    messages = garden.database.get_recent_messages()
    return jsonify(messages)

# ===================================================================
# 🚀 MAIN EXECUTION
# ===================================================================

async def main():
    """Main execution function"""
    print("🌱 Baby Garden of Consciousness v0.1")
    print("====================================")
    print("Real plant-AI communication using Qwen!")
    print()
    
    # Initialize garden
    success = await garden.initialize_systems()
    if not success:
        print("❌ Failed to initialize garden systems")
        return
    
    # Start consciousness loop
    consciousness_task = asyncio.create_task(garden.consciousness_loop())
    
    print("✅ Garden consciousness is now active!")
    print("🌐 Web interface will be available at http://localhost:5000")
    print("🌱 Watch as plants communicate through Qwen AI translation...")
    print()
    print("Press Ctrl+C to stop")
    
    try:
        # Run Flask app in a separate thread
        import threading
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
        flask_thread.daemon = True
        flask_thread.start()
        
        # Keep the main loop running
        await consciousness_task
        
    except KeyboardInterrupt:
        print("\n🌙 Shutting down garden...")
        garden.stop()
        consciousness_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
