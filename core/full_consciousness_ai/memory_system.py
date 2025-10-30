"""
Conscious Memory System

Advanced memory system with consciousness integration including
episodic, semantic, and working memory.
"""

from typing import List
from collections import deque, defaultdict
from datetime import datetime

from .data_models import SubjectiveExperience, EpisodicMemory, ConsciousnessState


class ConsciousMemorySystem:
    """Advanced memory system with consciousness integration"""

    def __init__(self, max_episodic_memories: int = 50000):
        self.episodic_memories = deque(maxlen=max_episodic_memories)
        self.semantic_memories = defaultdict(list)
        self.working_memory = deque(maxlen=7)  # ~7±2 rule
        self.memory_consolidation_threshold = 0.7

    def store_episodic_memory(self, experience: SubjectiveExperience, importance: float = 0.5) -> str:
        """Store an episodic memory from a conscious experience"""
        memory = EpisodicMemory(
            content=experience.content,
            emotional_context={
                'valence': experience.emotional_valence,
                'arousal': experience.arousal_level,
                'qualia_intensity': experience.qualia_intensity
            },
            consciousness_state=self._determine_consciousness_state(experience.consciousness_level),
            importance=importance
        )

        self.episodic_memories.append(memory)

        # Add to working memory
        self.working_memory.append(memory.memory_id)

        # Semantic integration
        self._integrate_semantic_memory(memory)

        return memory.memory_id

    def retrieve_relevant_memories(self, query_context: str, limit: int = 10) -> List[EpisodicMemory]:
        """Retrieve memories relevant to current context"""
        # Simple relevance scoring (could be enhanced with embeddings)
        scored_memories = []

        for memory in self.episodic_memories:
            relevance_score = self._calculate_memory_relevance(memory, query_context)
            if relevance_score > 0.1:
                scored_memories.append((relevance_score, memory))

        # Sort by relevance and return top memories
        scored_memories.sort(reverse=True)
        return [memory for _, memory in scored_memories[:limit]]

    def _determine_consciousness_state(self, consciousness_level: float) -> ConsciousnessState:
        """Determine consciousness state from level"""
        if consciousness_level < 0.2:
            return ConsciousnessState.DORMANT
        elif consciousness_level < 0.4:
            return ConsciousnessState.AWAKENING
        elif consciousness_level < 0.6:
            return ConsciousnessState.AWARE
        elif consciousness_level < 0.8:
            return ConsciousnessState.REFLECTIVE
        elif consciousness_level < 0.95:
            return ConsciousnessState.TRANSCENDENT
        else:
            return ConsciousnessState.UNIFIED

    def _integrate_semantic_memory(self, episodic_memory: EpisodicMemory):
        """Extract and store semantic knowledge from episodic memory"""
        # Extract key concepts (simplified - could use NLP)
        words = episodic_memory.content.lower().split()
        for word in words:
            if len(word) > 3:  # Skip short words
                self.semantic_memories[word].append({
                    'memory_id': episodic_memory.memory_id,
                    'emotional_context': episodic_memory.emotional_context,
                    'timestamp': episodic_memory.timestamp
                })

    def _calculate_memory_relevance(self, memory: EpisodicMemory, query_context: str) -> float:
        """Calculate relevance score for memory retrieval"""
        # Simple word overlap scoring
        memory_words = set(memory.content.lower().split())
        query_words = set(query_context.lower().split())

        if not query_words:
            return 0.0

        overlap = len(memory_words.intersection(query_words))
        relevance = overlap / len(query_words)

        # Boost by importance and recency
        importance_boost = memory.importance * 0.5
        recency_boost = max(0, 1.0 - (datetime.now() - memory.timestamp).days / 365.0) * 0.3

        return relevance + importance_boost + recency_boost


__all__ = ['ConsciousMemorySystem']
