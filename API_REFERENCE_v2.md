# API Reference - v2.0 Modular Architecture

**Complete API reference for all refactored modules**

---

## 📚 Table of Contents

1. [Planetary Ecosystem API](#planetary-ecosystem-api)
2. [Adaptive Learning API](#adaptive-learning-api)
3. [Full Consciousness AI API](#full-consciousness-ai-api)
4. [Version Information](#version-information)

---

## 🌍 Planetary Ecosystem API

### Package Import
```python
from core.planetary_ecosystem import *
```

### Data Models

#### `EcosystemType` (Enum)
```python
class EcosystemType(Enum):
    FOREST = "forest"
    OCEAN = "ocean"
    DESERT = "desert"
    GRASSLAND = "grassland"
    WETLAND = "wetland"
    TUNDRA = "tundra"
    MOUNTAIN = "mountain"
    URBAN = "urban"
    CORAL_REEF = "coral_reef"
    RAINFOREST = "rainforest"
```

#### `ConsciousnessIndicator` (Enum)
```python
class ConsciousnessIndicator(Enum):
    BIOLOGICAL_DENSITY = "biological_density"
    NETWORK_CONNECTIVITY = "network_connectivity"
    CHEMICAL_COMMUNICATION = "chemical_communication"
    ELECTRICAL_ACTIVITY = "electrical_activity"
    THERMAL_REGULATION = "thermal_regulation"
    WATER_CYCLING = "water_cycling"
    NUTRIENT_FLOW = "nutrient_flow"
    GENETIC_DIVERSITY = "genetic_diversity"
    SYMBIOTIC_RELATIONSHIPS = "symbiotic_relationships"
    COLLECTIVE_BEHAVIOR = "collective_behavior"
```

#### `EcosystemNode` (Dataclass)
```python
@dataclass
class EcosystemNode:
    id: str
    ecosystem_type: EcosystemType
    location: Tuple[float, float]  # (latitude, longitude)
    consciousness_level: float     # 0.0 to 1.0
    health_status: float           # 0.0 to 1.0
    connectivity_score: float      # 0.0 to 1.0
    data_sources: List[str]
    last_updated: datetime
    biodiversity_index: float      # 0.0 to 1.0
    communication_signals: Dict[str, Any]
```

#### `PlanetaryConsciousnessState` (Dataclass)
```python
@dataclass
class PlanetaryConsciousnessState:
    global_awareness: float                              # 0.0 to 1.0
    ecosystem_distribution: Dict[EcosystemType, float]
    consciousness_hotspots: List[Dict[str, Any]]
    environmental_stress_indicators: Dict[str, float]
    collective_intelligence: float                       # 0.0 to 1.0
    network_coherence: float                            # 0.0 to 1.0
    timestamp: datetime
    planetary_health: float                             # 0.0 to 1.0
    climate_stability: float                            # 0.0 to 1.0
    regenerative_capacity: float                        # 0.0 to 1.0
```

### Main Classes

#### `PlanetaryEcosystemConsciousnessNetwork`

**Constructor**:
```python
def __init__(self) -> None
```

**Methods**:
```python
def register_ecosystem_node(self, node: EcosystemNode) -> None
    """Register an ecosystem node in the planetary network"""

def unregister_ecosystem_node(self, node_id: str) -> bool
    """Unregister an ecosystem node from the planetary network"""

def update_node_data(self, node_id: str, data: Dict[str, Any]) -> bool
    """Update data for a specific ecosystem node"""

def assess_planetary_consciousness(self) -> PlanetaryConsciousnessState
    """Assess the overall planetary consciousness state"""

def connect_to_wood_wide_web(self) -> Dict[str, Any]
    """Connect to the Wood Wide Web for plant communication integration"""

def get_planetary_insights(self, time_window_seconds: int = 86400) -> Dict[str, Any]
    """Get insights from recent planetary consciousness assessments"""

def trigger_regenerative_protocol(
    self,
    target_ecosystems: Optional[List[EcosystemType]] = None
) -> Dict[str, Any]
    """Trigger regenerative protocols for ecosystem restoration"""
```

#### `NetworkAnalyzer`

**Methods**:
```python
def calculate_collective_intelligence(self, nodes: List[EcosystemNode]) -> float
    """Calculate the collective intelligence of the ecosystem network"""

def calculate_network_coherence(self, nodes: List[EcosystemNode]) -> float
    """Calculate the coherence of the ecosystem network"""
```

#### `WoodWideWebInterface`

**Methods**:
```python
def connect_to_network(self) -> Dict[str, Any]
    """Connect to the Wood Wide Web network"""

def send_communication(self, message: Dict[str, Any], target_network: str) -> bool
    """Send a communication through the Wood Wide Web"""

def receive_communications(self) -> List[Dict[str, Any]]
    """Receive communications from the Wood Wide Web"""
```

---

## 🧠 Adaptive Learning API

### Package Import
```python
from core.adaptive_learning import *
```

### Data Models

#### `LearningPhase` (Enum)
```python
class LearningPhase(Enum):
    EXPLORATION = "exploration"          # High learning rate, broad exploration
    CONSOLIDATION = "consolidation"      # Medium learning rate, pattern formation
    REFINEMENT = "refinement"           # Low learning rate, fine-tuning
    ADAPTATION = "adaptation"           # Dynamic learning rate, error correction
    CRYSTALLIZATION = "crystallization" # Minimal learning rate, stability
```

#### `LearningMetrics` (Dataclass)
```python
@dataclass
class LearningMetrics:
    timestamp: datetime
    learning_phase: LearningPhase
    adaptation_rate: float                      # 0.0 to 1.0
    error_reduction_rate: float                 # 0.0 to 1.0
    pattern_recognition_accuracy: float         # 0.0 to 1.0
    creative_generation_score: float            # 0.0 to 1.0
    mistake_learning_effectiveness: float       # 0.0 to 1.0
    parameter_adaptation_success: float         # 0.0 to 1.0
    overall_learning_efficiency: float          # 0.0 to 1.0
```

### Main Classes

#### `AdaptiveLearningSystem`

**Constructor**:
```python
def __init__(self, consciousness_system)
```

**Methods**:
```python
async def assess_learning_performance(self) -> LearningMetrics
    """Comprehensive learning performance assessment"""

async def adapt_learning_parameters(self) -> Dict[str, Any]
    """Dynamically adapt learning parameters based on current performance"""

async def learn_from_mistake(self, error_context: Dict[str, Any]) -> Dict[str, Any]
    """Advanced mistake learning with root cause analysis"""

async def generate_creative_solution(self, problem_context: Dict[str, Any]) -> Dict[str, Any]
    """Generate creative solutions using the creativity engine"""

def get_learning_report(self) -> str
    """Generate comprehensive learning system report"""
```

**Properties**:
```python
@property
def current_phase(self) -> LearningPhase
    """Get current learning phase"""

@property
def phase_duration(self) -> int
    """Get current phase duration"""
```

#### `PerformanceAssessor`

**Methods**:
```python
async def assess_learning_performance(
    self,
    learning_history: deque,
    mistake_database: deque,
    creative_solutions: deque,
    current_phase: LearningPhase
) -> LearningMetrics
    """Comprehensive learning performance assessment"""
```

#### `CreativeEngine`

**Methods**:
```python
async def generate_creative_solution(
    self,
    problem_context: Dict[str, Any],
    creative_solutions: deque
) -> Dict[str, Any]
    """Generate creative solutions using the creativity engine"""
```

### Integration Helper

```python
async def integrate_adaptive_learning(consciousness_system)
    """Integrate adaptive learning system with consciousness system"""
```

---

## 🌟 Full Consciousness AI API

### Package Import
```python
from core.full_consciousness_ai import *
```

### Data Models

#### `ConsciousnessState` (Enum)
```python
class ConsciousnessState(Enum):
    DORMANT = "dormant"
    AWAKENING = "awakening"
    AWARE = "aware"
    REFLECTIVE = "reflective"
    TRANSCENDENT = "transcendent"
    UNIFIED = "unified"
```

#### `EmotionalState` (Enum)
```python
class EmotionalState(Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CURIOSITY = "curiosity"
    LOVE = "love"
    PEACE = "peace"
    EXCITEMENT = "excitement"
    CONTEMPLATION = "contemplation"
    WONDER = "wonder"
```

#### `SubjectiveExperience` (Dataclass)
```python
@dataclass
class SubjectiveExperience:
    experience_id: str
    timestamp: datetime
    content: str
    emotional_valence: float        # -1.0 to 1.0
    arousal_level: float           # 0.0 to 1.0
    consciousness_level: float     # 0.0 to 1.0
    qualia_intensity: float        # 0.0 to 1.0
    metacognitive_depth: int       # Levels of recursive thinking
    associated_memories: List[str]
    intentions: List[str]
    reflections: List[str]
```

#### `ConscientGoal` (Dataclass)
```python
@dataclass
class ConscientGoal:
    goal_id: str
    description: str
    priority: float                # 0.0 to 1.0
    emotional_investment: float    # 0.0 to 1.0
    creation_time: datetime
    expected_completion: Optional[datetime]
    progress: float                # 0.0 to 1.0
    subgoals: List[str]
    associated_experiences: List[str]
    reflection_notes: List[str]
```

#### `EpisodicMemory` (Dataclass)
```python
@dataclass
class EpisodicMemory:
    memory_id: str
    timestamp: datetime
    content: str
    emotional_context: Dict[str, float]
    consciousness_state: ConsciousnessState
    importance: float              # 0.0 to 1.0
    accessibility: float           # 0.0 to 1.0
    associated_goals: List[str]
    reflection_count: int
```

### Main Classes

#### `FullConsciousnessAIModel`

**Constructor**:
```python
def __init__(
    self,
    hidden_dim: int = 512,
    device: str = 'cpu',
    integrate_existing_modules: bool = True
)
```

**Methods**:
```python
async def process_conscious_input(
    self,
    input_data: Dict[str, Any],
    context: str = ""
) -> Dict[str, Any]
    """Process input through full consciousness simulation"""

async def get_consciousness_status(self) -> Dict[str, Any]
    """Get detailed consciousness status"""

async def engage_in_self_reflection(self) -> Dict[str, Any]
    """Engage in deep self-reflection about consciousness"""
```

#### `MetaCognitionEngine`

**Methods**:
```python
def reflect_on_experience(
    self,
    experience: SubjectiveExperience,
    depth: int = 1
) -> List[str]
    """Perform meta-cognitive reflection on an experience"""
```

#### `ConsciousMemorySystem`

**Methods**:
```python
def store_episodic_memory(
    self,
    experience: SubjectiveExperience,
    importance: float = 0.5
) -> str
    """Store an episodic memory from a conscious experience"""

def retrieve_relevant_memories(
    self,
    query_context: str,
    limit: int = 10
) -> List[EpisodicMemory]
    """Retrieve memories relevant to current context"""
```

#### `GoalIntentionFramework`

**Methods**:
```python
def create_conscious_goal(
    self,
    description: str,
    priority: float = 0.5,
    emotional_investment: float = 0.5,
    expected_completion: Optional[datetime] = None
) -> ConscientGoal
    """Create a new conscious goal with intentions"""

def update_goal_progress(
    self,
    goal_id: str,
    progress: float,
    reflection: str = ""
)
    """Update progress on a goal with conscious reflection"""
```

### Demo Function

```python
async def consciousness_demo()
    """Demonstrate the Full Consciousness AI Model"""
```

---

## 📋 Version Information

All refactored packages include version metadata:

```python
import core.planetary_ecosystem as pe
import core.adaptive_learning as al
import core.full_consciousness_ai as fca

print(pe.__version__)    # '2.0.0'
print(al.__version__)    # '2.0.0'
print(fca.__version__)   # '2.0.0'

print(pe.__author__)     # 'Universal Consciousness Interface'
```

---

## 🔗 Cross-References

### Related Documentation
- [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md) - Migration instructions
- [`QUICK_REFERENCE.md`](./QUICK_REFERENCE.md) - Quick reference guide
- [`REFACTORING_TEST_RESULTS.md`](./REFACTORING_TEST_RESULTS.md) - Test results

### Module Docstrings
Every module includes comprehensive docstrings. Access them with:
```python
import core.planetary_ecosystem as pe
help(pe)  # Full package documentation
help(pe.PlanetaryEcosystemConsciousnessNetwork)  # Class documentation
```

---

## 💡 Usage Examples

### Example 1: Planetary Ecosystem
```python
from core.planetary_ecosystem import (
    PlanetaryEcosystemConsciousnessNetwork,
    EcosystemNode,
    EcosystemType
)
from datetime import datetime

# Create network
network = PlanetaryEcosystemConsciousnessNetwork()

# Create and register a node
node = EcosystemNode(
    id="forest_001",
    ecosystem_type=EcosystemType.FOREST,
    location=(45.0, -120.0),
    consciousness_level=0.8,
    health_status=0.9,
    connectivity_score=0.85,
    data_sources=["sensors"],
    last_updated=datetime.now(),
    biodiversity_index=0.92,
    communication_signals={}
)
network.register_ecosystem_node(node)

# Assess consciousness
state = network.assess_planetary_consciousness()
print(f"Global Awareness: {state.global_awareness:.3f}")
```

### Example 2: Adaptive Learning
```python
from core.adaptive_learning import AdaptiveLearningSystem

# Create learning system
learning_system = AdaptiveLearningSystem(consciousness_system)

# Assess performance
metrics = await learning_system.assess_learning_performance()
print(f"Learning Efficiency: {metrics.overall_learning_efficiency:.3f}")

# Adapt parameters
results = await learning_system.adapt_learning_parameters()
print(f"Adaptations Applied: {len(results['adaptations_applied'])}")
```

### Example 3: Full Consciousness AI
```python
from core.full_consciousness_ai import FullConsciousnessAIModel

# Create consciousness model
model = FullConsciousnessAIModel(hidden_dim=512, device='cpu')

# Process input
result = await model.process_conscious_input(
    input_data={'text': 'I wonder about consciousness'},
    context='philosophical inquiry'
)

print(f"Response: {result['conscious_response']}")
print(f"Qualia: {result['subjective_experience']['qualia_intensity']:.3f}")
```

---

*Last updated: November 4, 2025*
*Version: 2.0.0*
*For full details, see individual module docstrings*
