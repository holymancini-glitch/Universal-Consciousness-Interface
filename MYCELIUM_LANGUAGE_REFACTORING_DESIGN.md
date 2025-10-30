# Mycelium Language Generator Refactoring Design

**Date:** 2025-10-30
**Original File:** `core/mycelium_language_generator.py` (1,149 lines)
**Target:** Break into 6 specialized modules
**Status:** Phase 2 - Planning

---

## Current Structure Analysis

### File Statistics:
- **Total Lines:** 1,149
- **Classes:** 5 (4 dataclasses + 1 main generator)
- **Methods in MyceliumLanguageGenerator:** 46
- **Imports:** logging, dataclasses, enum, datetime, random, defaultdict, deque, List, Dict, Any, Optional, Tuple, Union, asyncio, numpy (optional)

### Class Breakdown:
1. `MyceliumCommunicationType` (Enum) - Line 44
2. `MyceliumSignal` (dataclass) - Line 55
3. `MyceliumWord` (dataclass) - Line 67
4. `MyceliumSentence` (dataclass) - Line 77
5. `MyceliumLanguageGenerator` (main class) - Line 86

---

## Method Categorization (46 methods)

### Category 1: Initialization (6 methods)
- `__init__()` - Main initialization
- `_initialize_communication_patterns()` - Lines 131-175 (45 lines)
- `_initialize_phonetic_library()` - Lines 177-210 (34 lines)
- `_initialize_chemical_vocabulary()` - Lines 212-245 (34 lines)
- `_initialize_syntactic_rules()` - Lines 247-285 (39 lines)
- `_initialize_network_topology()` - Lines 287-296 (10 lines)

### Category 2: Signal Processing (6 methods)
- `_process_signals_to_tokens()` - Lines 336-389 (54 lines)
- `_chemical_to_phonetic()` - Lines 391-411 (21 lines)
- `_frequency_to_phonetic()` - Lines 413-422 (10 lines)
- `_flow_to_phonetic()` - Lines 424-441 (18 lines)
- `_resonance_to_phonetic()` - Lines 443-453 (11 lines)
- `_generic_to_phonetic()` - Lines 455-457 (3 lines)

### Category 3: Word Generation (6 methods)
- `_generate_words_from_patterns()` - Lines 459-494 (36 lines)
- `_group_tokens_semantically()` - Lines 496-514 (19 lines)
- `_combine_phonetic_patterns()` - Lines 516-523 (8 lines)
- `_generate_chemical_signature()` - Lines 525-548 (24 lines)
- `_calculate_electrical_signature()` - Lines 550-565 (16 lines)
- `_derive_meaning_concept()` - Lines 567-590 (24 lines)

### Category 4: Syntactic Structure (7 methods)
- `_generate_syntactic_structure()` - Lines 592-604 (13 lines)
- `_determine_word_order()` - Lines 606-615 (10 lines)
- `_determine_phrase_structure()` - Lines 617-626 (10 lines)
- `_map_semantic_relations()` - Lines 628-647 (20 lines)
- `_determine_temporal_flow()` - Lines 649-658 (10 lines)
- `_calculate_semantic_similarity()` - Lines 660-672 (13 lines)
- `_calculate_chemical_affinity()` - Lines 674-688 (15 lines)

### Category 5: Sentence Assembly (7 methods)
- `_assemble_sentences()` - Lines 690-725 (36 lines)
- `_group_words_into_sentences()` - Lines 727-747 (21 lines)
- `_order_words_in_sentence()` - Lines 749-772 (24 lines)
- `_determine_sentence_consciousness()` - Lines 774-791 (18 lines)
- `_generate_semantic_flow()` - Lines 793-801 (9 lines)
- `_trace_chemical_flow()` - Lines 803-811 (9 lines)
- `_trace_electrical_flow()` - Lines 813-815 (3 lines)
- `_calculate_sentence_coherence()` - Lines 817-830 (14 lines)

### Category 6: Language Evolution (9 methods)
- `_evolve_language_patterns()` - Lines 832-850 (19 lines)
- `_generate_pattern_mutations()` - Lines 852-875 (24 lines)
- `_mutate_phonetic_pattern()` - Lines 877-889 (13 lines)
- `_mutate_syntactic_structure()` - Lines 891-900 (10 lines)
- `_calculate_semantic_drift()` - Lines 902-915 (14 lines)
- `_identify_network_adaptations()` - Lines 917-937 (21 lines)
- `_detect_consciousness_emergence()` - Lines 939-957 (19 lines)
- `_identify_novel_constructions()` - Lines 959-983 (25 lines)
- `_update_language_metrics()` - Lines 985-997 (13 lines)

### Category 7: Utility/Demo (3 methods)
- `get_language_summary()` - Lines 999-1013 (15 lines)
- `demonstrate_mycelium_language_generation()` - Lines 1015-1055 (41 lines)
- `generate_sample_signals()` - Lines 1057-1110 (54 lines)

---

## Proposed Module Structure

### New Directory Structure:
```
core/
├── mycelium_language_generator.py       (150-200 lines - main facade)
└── mycelium_language/
    ├── __init__.py                      (50 lines - exports)
    ├── data_models.py                   (80 lines - dataclasses & enums)
    ├── vocabulary_manager.py            (200 lines - initialization)
    ├── network_processor.py             (250 lines - network & signals)
    ├── biochemical_translator.py        (220 lines - signal→phonetic)
    ├── language_synthesizer.py          (300 lines - word & sentence generation)
    ├── pattern_analyzer.py              (200 lines - syntactic & semantic analysis)
    └── evolution_engine.py              (180 lines - language evolution)
```

---

## Module 1: data_models.py (~80 lines)

**Purpose:** Centralize all data structures

**Contents:**
```python
# Enums
- MyceliumCommunicationType (enum with 5 types)

# Dataclasses
- MyceliumSignal
- MyceliumWord
- MyceliumSentence

# Type aliases
- SignalList = List[MyceliumSignal]
- WordList = List[MyceliumWord]
- SentenceList = List[MyceliumSentence]
```

**Dependencies:** None (pure data)

**Imports:** dataclasses, enum, datetime, typing

---

## Module 2: vocabulary_manager.py (~200 lines)

**Purpose:** Manage phonetic, chemical, and syntactic vocabularies

**Classes:**
```python
class VocabularyManager:
    def __init__(self):
        self.phonetic_library: Dict[str, str]
        self.chemical_vocabulary: Dict[str, Dict[str, float]]
        self.syntactic_rules: Dict[str, List[str]]
        self.communication_patterns: Dict[str, Any]
```

**Methods:**
- `initialize_phonetic_library()` → Dict[str, str]
- `initialize_chemical_vocabulary()` → Dict[str, Dict[str, float]]
- `initialize_syntactic_rules()` → Dict[str, List[str]]
- `initialize_communication_patterns()` → Dict[str, Any]
- `get_random_phoneme()` → str
- `get_chemical_vocab(key)` → Dict[str, float]

**Dependencies:**
- data_models.MyceliumCommunicationType

**Estimated Lines:** ~200
- Phonetic library: ~40 lines
- Chemical vocabulary: ~40 lines
- Syntactic rules: ~45 lines
- Communication patterns: ~50 lines
- Helper methods: ~25 lines

---

## Module 3: network_processor.py (~250 lines)

**Purpose:** Network topology and signal generation

**Classes:**
```python
class NetworkProcessor:
    def __init__(self, network_size: int = 1000):
        self.network_size = network_size
        self.network_topology: Dict[str, Any]
        self.active_signals: Deque[MyceliumSignal]
```

**Methods:**
- `initialize_network_topology()` → Dict[str, Any]
- `generate_sample_signals(count: int)` → List[MyceliumSignal]
- `add_signal(signal: MyceliumSignal)` → None
- `get_topology_metrics()` → Dict[str, float]
- `analyze_network_connectivity()` → Dict[str, Any]
- `determine_word_order(topology)` → str
- `determine_phrase_structure(topology)` → str
- `determine_temporal_flow(topology)` → str

**Dependencies:**
- data_models.MyceliumSignal, MyceliumCommunicationType

**Estimated Lines:** ~250
- Network initialization: ~15 lines
- Sample signal generation: ~60 lines
- Topology analysis: ~80 lines
- Word order determination: ~15 lines
- Phrase structure: ~15 lines
- Temporal flow: ~15 lines
- Helper methods: ~50 lines

---

## Module 4: biochemical_translator.py (~220 lines)

**Purpose:** Translate signals to phonetic patterns

**Classes:**
```python
class BiochemicalTranslator:
    def __init__(self, vocabulary_manager: VocabularyManager):
        self.vocabulary = vocabulary_manager
```

**Methods:**
- `process_signals_to_tokens(signals)` → List[Dict[str, Any]]
- `chemical_to_phonetic(composition)` → str
- `frequency_to_phonetic(frequency)` → str
- `flow_to_phonetic(signal)` → str
- `resonance_to_phonetic(signal)` → str
- `generic_to_phonetic(signal)` → str
- `combine_phonetic_patterns(patterns)` → str
- `generate_chemical_signature(tokens)` → Dict[str, float]
- `calculate_electrical_signature(tokens)` → float
- `derive_meaning_concept(tokens, consciousness_level)` → str

**Dependencies:**
- data_models.MyceliumSignal, MyceliumCommunicationType
- vocabulary_manager.VocabularyManager

**Estimated Lines:** ~220
- Token processing: ~60 lines
- Phonetic conversion methods: ~60 lines
- Signature generation: ~60 lines
- Meaning derivation: ~30 lines
- Helper methods: ~10 lines

---

## Module 5: language_synthesizer.py (~300 lines)

**Purpose:** Generate words and assemble sentences

**Classes:**
```python
class LanguageSynthesizer:
    def __init__(self, translator: BiochemicalTranslator,
                 pattern_analyzer: PatternAnalyzer):
        self.translator = translator
        self.pattern_analyzer = pattern_analyzer
        self.mycelium_words: List[MyceliumWord] = []
        self.mycelium_sentences: List[MyceliumSentence] = []
```

**Methods:**
- `generate_words_from_patterns(tokens, consciousness_level)` → List[MyceliumWord]
- `group_tokens_semantically(tokens)` → Dict[str, List[Dict]]
- `assemble_sentences(words, structure)` → List[MyceliumSentence]
- `group_words_into_sentences(words, relations)` → List[List[MyceliumWord]]
- `order_words_in_sentence(words, word_order)` → List[MyceliumWord]
- `determine_sentence_consciousness(words)` → str
- `generate_semantic_flow(words, relations)` → Dict[str, Any]
- `trace_chemical_flow(words)` → Dict[str, List[float]]
- `trace_electrical_flow(words)` → List[float]
- `calculate_sentence_coherence(words)` → float

**Dependencies:**
- data_models.MyceliumWord, MyceliumSentence
- biochemical_translator.BiochemicalTranslator
- pattern_analyzer.PatternAnalyzer

**Estimated Lines:** ~300
- Word generation: ~50 lines
- Semantic grouping: ~25 lines
- Sentence assembly: ~40 lines
- Word grouping: ~25 lines
- Word ordering: ~30 lines
- Consciousness determination: ~20 lines
- Semantic flow: ~30 lines
- Flow tracing: ~30 lines
- Coherence calculation: ~20 lines
- Helper methods: ~30 lines

---

## Module 6: pattern_analyzer.py (~200 lines)

**Purpose:** Analyze syntactic and semantic patterns

**Classes:**
```python
class PatternAnalyzer:
    def __init__(self, network_processor: NetworkProcessor):
        self.network = network_processor
```

**Methods:**
- `generate_syntactic_structure(words)` → Dict[str, Any]
- `map_semantic_relations(words)` → Dict[str, List[str]]
- `calculate_semantic_similarity(word1, word2)` → float
- `calculate_chemical_affinity(word1, word2)` → float
- `analyze_word_complexity(word)` → float
- `detect_pattern_clusters(words)` → List[List[MyceliumWord]]
- `measure_linguistic_coherence(words)` → float

**Dependencies:**
- data_models.MyceliumWord
- network_processor.NetworkProcessor

**Estimated Lines:** ~200
- Syntactic structure: ~20 lines
- Semantic relations: ~30 lines
- Similarity calculation: ~20 lines
- Chemical affinity: ~20 lines
- Complexity analysis: ~30 lines
- Pattern clustering: ~40 lines
- Coherence measurement: ~20 lines
- Helper methods: ~20 lines

---

## Module 7: evolution_engine.py (~180 lines)

**Purpose:** Evolve language patterns and track emergence

**Classes:**
```python
class EvolutionEngine:
    def __init__(self):
        self.language_evolution_history: List[Dict[str, Any]] = []
        self.linguistic_complexity: float = 0.0
        self.semantic_coherence: float = 0.0
        self.novel_language_count: int = 0
```

**Methods:**
- `evolve_language_patterns(sentences)` → Dict[str, Any]
- `generate_pattern_mutations(sentences)` → List[Dict[str, Any]]
- `mutate_phonetic_pattern(pattern)` → str
- `mutate_syntactic_structure(structure)` → str
- `calculate_semantic_drift(sentences)` → Dict[str, Union[float, str, int]]
- `identify_network_adaptations(sentences)` → List[str]
- `detect_consciousness_emergence(sentences)` → Dict[str, Any]
- `identify_novel_constructions(sentences)` → List[Dict[str, Any]]
- `update_language_metrics(evolved_language)` → None

**Dependencies:**
- data_models.MyceliumSentence, MyceliumWord

**Estimated Lines:** ~180
- Pattern evolution: ~25 lines
- Mutation generation: ~30 lines
- Phonetic mutation: ~15 lines
- Syntactic mutation: ~15 lines
- Semantic drift: ~20 lines
- Network adaptations: ~25 lines
- Consciousness detection: ~25 lines
- Novel constructions: ~30 lines

---

## Module 8: mycelium_language_generator.py (Main Facade) (~180 lines)

**Purpose:** Main entry point - coordinates all modules

**Classes:**
```python
class MyceliumLanguageGenerator:
    def __init__(self, network_size: int = 1000):
        # Initialize all sub-components
        self.vocabulary_manager = VocabularyManager()
        self.network_processor = NetworkProcessor(network_size)
        self.biochemical_translator = BiochemicalTranslator(self.vocabulary_manager)
        self.pattern_analyzer = PatternAnalyzer(self.network_processor)
        self.language_synthesizer = LanguageSynthesizer(
            self.biochemical_translator,
            self.pattern_analyzer
        )
        self.evolution_engine = EvolutionEngine()

        # Consciousness mapping
        self.consciousness_mapping = {...}
```

**Methods:**
- `async generate_mycelium_language(signals, consciousness_level)` → Dict[str, Any]
- `get_language_summary()` → Dict[str, Any]
- `async demonstrate_mycelium_language_generation()` → Dict[str, Any]

**Property Accessors (for backward compatibility):**
- `@property phonetic_library` → delegates to vocabulary_manager
- `@property chemical_vocabulary` → delegates to vocabulary_manager
- `@property network_topology` → delegates to network_processor
- `@property mycelium_words` → delegates to language_synthesizer
- `@property mycelium_sentences` → delegates to language_synthesizer

**Dependencies:** All 7 specialized modules

**Estimated Lines:** ~180
- Initialization: ~30 lines
- Main generation method: ~40 lines
- Summary method: ~20 lines
- Demo method: ~50 lines
- Property accessors: ~40 lines

---

## Module 9: __init__.py (~50 lines)

**Purpose:** Public API exports and convenience imports

**Contents:**
```python
# Import all data models
from .data_models import (
    MyceliumCommunicationType,
    MyceliumSignal,
    MyceliumWord,
    MyceliumSentence
)

# Import main classes
from .vocabulary_manager import VocabularyManager
from .network_processor import NetworkProcessor
from .biochemical_translator import BiochemicalTranslator
from .language_synthesizer import LanguageSynthesizer
from .pattern_analyzer import PatternAnalyzer
from .evolution_engine import EvolutionEngine

# Public API
__all__ = [
    'MyceliumCommunicationType',
    'MyceliumSignal',
    'MyceliumWord',
    'MyceliumSentence',
    'VocabularyManager',
    'NetworkProcessor',
    'BiochemicalTranslator',
    'LanguageSynthesizer',
    'PatternAnalyzer',
    'EvolutionEngine'
]
```

---

## Dependency Graph

```
mycelium_language_generator.py (main facade)
├── vocabulary_manager.py
│   └── data_models.py
├── network_processor.py
│   └── data_models.py
├── biochemical_translator.py
│   ├── data_models.py
│   └── vocabulary_manager.py
├── pattern_analyzer.py
│   ├── data_models.py
│   └── network_processor.py
├── language_synthesizer.py
│   ├── data_models.py
│   ├── biochemical_translator.py
│   └── pattern_analyzer.py
└── evolution_engine.py
    └── data_models.py
```

**No Circular Dependencies** ✅

---

## Backward Compatibility Strategy

### Original Import Path Preserved:
```python
# This will still work
from core.mycelium_language_generator import MyceliumLanguageGenerator

# New modular imports also available
from core.mycelium_language import (
    VocabularyManager,
    NetworkProcessor,
    BiochemicalTranslator
)
```

### Property Delegation:
The main `MyceliumLanguageGenerator` class will expose properties that delegate to sub-modules:
```python
@property
def phonetic_library(self):
    return self.vocabulary_manager.phonetic_library

@property
def mycelium_words(self):
    return self.language_synthesizer.mycelium_words
```

---

## Implementation Order

### Step 1: Create data_models.py
- Move 4 dataclasses + 1 enum
- No dependencies
- Quick win: ~30 minutes

### Step 2: Create vocabulary_manager.py
- Move initialization methods
- Import from data_models
- Estimated time: 1 hour

### Step 3: Create network_processor.py
- Move network topology methods
- Import from data_models
- Estimated time: 1.5 hours

### Step 4: Create biochemical_translator.py
- Move signal processing methods
- Import vocabulary_manager
- Estimated time: 1.5 hours

### Step 5: Create pattern_analyzer.py
- Move syntactic/semantic analysis
- Import network_processor
- Estimated time: 1 hour

### Step 6: Create language_synthesizer.py
- Move word/sentence generation
- Import biochemical_translator & pattern_analyzer
- Estimated time: 2 hours

### Step 7: Create evolution_engine.py
- Move evolution methods
- Import data_models
- Estimated time: 1 hour

### Step 8: Refactor mycelium_language_generator.py
- Create facade with delegation
- Add property accessors
- Estimated time: 1.5 hours

### Step 9: Create __init__.py
- Export public API
- Estimated time: 15 minutes

### Step 10: Testing
- Run existing tests
- Add new module tests
- Estimated time: 2 hours

**Total Estimated Time:** 12-14 hours

---

## Success Criteria

### Code Quality:
- ✅ No file >300 lines (target: <250 for most)
- ✅ Clear single responsibility per module
- ✅ No circular dependencies
- ✅ All imports explicit and clean

### Functionality:
- ✅ All existing tests pass
- ✅ Backward compatible imports work
- ✅ No breaking changes to public API

### Documentation:
- ✅ Each module has comprehensive docstrings
- ✅ Dependency graph documented
- ✅ Migration examples provided

---

## Risk Assessment

### Low Risk:
- Data models extraction (no logic)
- Vocabulary manager (pure initialization)

### Medium Risk:
- Language synthesizer (complex dependencies)
- Main facade refactoring (needs careful delegation)

### Mitigation:
- Test after each module creation
- Commit frequently
- Keep original file until all tests pass

---

**Status:** Ready for Phase 3 Implementation
**Next Action:** Create core/mycelium_language/ directory and begin with data_models.py

---

Generated as part of mycelium_language_generator.py refactoring (Phase 2: Planning)
