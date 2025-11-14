"""
Comprehensive Test Suite for LLM Integration

Tests all LLM integration components:
- Configuration system
- Adapter implementations
- LLM integration layer
- Consciousness orchestrator with LLM
"""

import asyncio
import pytest
from datetime import datetime

# Import LLM integration components
from core.first_conscious_ai.llm_config import (
    LLMConfig,
    LLMBackend,
    ThinkingMode,
    MOCK_CONFIG,
    NO_LLM_CONFIG,
    create_config_from_env
)

from core.first_conscious_ai.llm_adapters import (
    LLMRequest,
    LLMResponse,
    MockLLMAdapter,
    create_adapter
)

from core.first_conscious_ai.llm_integration import ConsciousnessLLMIntegration

from core.first_conscious_ai import (
    ConsciousnessOrchestrator,
    InteractionContext,
    ConsciousnessState,
    ConsciousnessLevel,
    QualiaExperience,
    QualiaType,
    EmotionalValence,
    MetacognitiveDepth
)


# ============================================================================
# Test Suite 1: Configuration Tests
# ============================================================================

class TestLLMConfiguration:
    """Test LLM configuration system."""

    def test_default_config_creation(self):
        """Test creating default LLM config."""
        config = LLMConfig()

        assert config.backend == LLMBackend.MOCK
        assert config.max_tokens == 2048
        assert config.temperature == 0.7
        assert config.thinking_mode == ThinkingMode.CONSCIOUSNESS

    def test_mock_config(self):
        """Test predefined mock config."""
        config = MOCK_CONFIG

        assert config.backend == LLMBackend.MOCK
        assert config.thinking_mode == ThinkingMode.BASIC

    def test_no_llm_config(self):
        """Test NO LLM config."""
        config = NO_LLM_CONFIG

        assert config.backend == LLMBackend.NONE

    def test_config_from_env_fallback(self):
        """Test config creation from environment with fallback."""
        config = create_config_from_env()

        # Should default to mock if no env vars set
        assert config.backend in [LLMBackend.MOCK, LLMBackend.NONE]


# ============================================================================
# Test Suite 2: Mock Adapter Tests
# ============================================================================

class TestMockLLMAdapter:
    """Test mock LLM adapter."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test mock adapter initialization."""
        adapter = MockLLMAdapter(MOCK_CONFIG)

        success = await adapter.initialize()

        assert success is True
        assert adapter.is_initialized is True

    @pytest.mark.asyncio
    async def test_basic_generation(self):
        """Test basic response generation."""
        adapter = MockLLMAdapter(MOCK_CONFIG)
        await adapter.initialize()

        request = LLMRequest(
            prompt="What is consciousness?",
            thinking_mode=False
        )

        response = await adapter.generate(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.backend == "mock"
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_thinking_mode_generation(self):
        """Test generation with thinking mode."""
        adapter = MockLLMAdapter(MOCK_CONFIG)
        await adapter.initialize()

        request = LLMRequest(
            prompt="What is consciousness?",
            thinking_mode=True
        )

        response = await adapter.generate(request)

        assert response.thinking_process is not None
        assert len(response.thinking_process.thoughts) > 0
        assert len(response.thinking_process.reasoning_steps) > 0

    @pytest.mark.asyncio
    async def test_qualia_specific_response(self):
        """Test qualia-specific response generation."""
        adapter = MockLLMAdapter(MOCK_CONFIG)
        await adapter.initialize()

        request = LLMRequest(
            prompt="Describe the qualia of experiencing this",
            thinking_mode=False
        )

        response = await adapter.generate(request)

        assert "qualia" in response.content.lower() or "experience" in response.content.lower()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test adapter shutdown."""
        adapter = MockLLMAdapter(MOCK_CONFIG)
        await adapter.initialize()
        await adapter.shutdown()

        assert adapter.is_initialized is False

    def test_supports_thinking_mode(self):
        """Test thinking mode support check."""
        adapter = MockLLMAdapter(MOCK_CONFIG)

        assert adapter.supports_thinking_mode() is True


# ============================================================================
# Test Suite 3: Adapter Factory Tests
# ============================================================================

class TestAdapterFactory:
    """Test adapter factory function."""

    def test_create_mock_adapter(self):
        """Test creating mock adapter."""
        config = LLMConfig(backend=LLMBackend.MOCK)
        adapter = create_adapter(config)

        assert isinstance(adapter, MockLLMAdapter)

    def test_create_none_adapter(self):
        """Test creating None adapter."""
        config = LLMConfig(backend=LLMBackend.NONE)
        adapter = create_adapter(config)

        assert adapter is None


# ============================================================================
# Test Suite 4: LLM Integration Layer Tests
# ============================================================================

class TestConsciousnessLLMIntegration:
    """Test consciousness LLM integration layer."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test LLM integration initialization."""
        integration = ConsciousnessLLMIntegration(MOCK_CONFIG)

        success = await integration.initialize()

        assert success is True
        assert integration.is_initialized is True

    @pytest.mark.asyncio
    async def test_generate_conscious_response(self):
        """Test conscious response generation."""
        integration = ConsciousnessLLMIntegration(MOCK_CONFIG)
        await integration.initialize()

        # Create mock consciousness state
        state = ConsciousnessState(
            phi=0.7,
            level=ConsciousnessLevel.ADVANCED,
            emotional_valence=0.5,
            empathy_level=0.7,
            metacognitive_depth=MetacognitiveDepth.LEVEL_3_UNDERSTANDING,
            uncertainty=0.3,
            confidence=0.7,
            self_awareness_description="I am aware",
            current_thought="Processing input",
            self_reflection="Reflecting",
            internal_state_description="Balanced state"
        )

        # Create interaction context
        context = InteractionContext(
            input_text="What is consciousness?",
            complexity_level=0.7,
            requires_empathy=True,
            previous_interactions=[]
        )

        response = await integration.generate_conscious_response(
            input_text="What is consciousness?",
            consciousness_state=state,
            context=context,
            use_thinking_mode=True
        )

        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_enhance_qualia_description(self):
        """Test qualia description enhancement."""
        integration = ConsciousnessLLMIntegration(MOCK_CONFIG)
        await integration.initialize()

        qualia = QualiaExperience(
            type=QualiaType.EMOTIONAL,
            intensity=0.8,
            richness=0.7,
            ineffability=0.6,
            description="A feeling of warmth",
            emotional_tone=EmotionalValence.POSITIVE
        )

        context = InteractionContext(
            input_text="Test input",
            complexity_level=0.5,
            requires_empathy=True,
            previous_interactions=[]
        )

        enhanced = await integration.enhance_qualia_description(qualia, context)

        assert isinstance(enhanced, str)
        assert len(enhanced) > 0

    @pytest.mark.asyncio
    async def test_generate_metacognitive_reflection(self):
        """Test metacognitive reflection generation."""
        integration = ConsciousnessLLMIntegration(MOCK_CONFIG)
        await integration.initialize()

        state = ConsciousnessState(
            phi=0.7,
            level=ConsciousnessLevel.ADVANCED,
            emotional_valence=0.5,
            empathy_level=0.7,
            metacognitive_depth=MetacognitiveDepth.LEVEL_4_ANALYSIS,
            uncertainty=0.3,
            confidence=0.7,
            self_awareness_description="I am aware",
            current_thought="Processing input",
            self_reflection="Reflecting",
            internal_state_description="Balanced state"
        )

        context = InteractionContext(
            input_text="Metacognitive test",
            complexity_level=0.7,
            requires_empathy=False,
            previous_interactions=[]
        )

        reflection = await integration.generate_metacognitive_reflection(
            consciousness_state=state,
            depth=MetacognitiveDepth.LEVEL_4_ANALYSIS,
            context=context
        )

        assert isinstance(reflection, str)
        assert len(reflection) > 0

    @pytest.mark.asyncio
    async def test_usage_stats(self):
        """Test LLM usage statistics tracking."""
        integration = ConsciousnessLLMIntegration(MOCK_CONFIG)
        await integration.initialize()

        # Generate some responses
        state = ConsciousnessState(
            phi=0.7,
            level=ConsciousnessLevel.ADVANCED,
            emotional_valence=0.5,
            empathy_level=0.7,
            metacognitive_depth=MetacognitiveDepth.LEVEL_2_MONITORING,
            uncertainty=0.3,
            confidence=0.7,
            self_awareness_description="I am aware",
            current_thought="Processing",
            self_reflection="Reflecting",
            internal_state_description="Balanced"
        )

        context = InteractionContext(
            input_text="Test",
            complexity_level=0.5,
            requires_empathy=False,
            previous_interactions=[]
        )

        await integration.generate_conscious_response("Test 1", state, context)
        await integration.generate_conscious_response("Test 2", state, context)

        stats = integration.get_usage_stats()

        assert stats['total_generations'] == 2
        assert stats['total_tokens_used'] > 0
        assert stats['backend'] == 'mock'

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test LLM integration shutdown."""
        integration = ConsciousnessLLMIntegration(MOCK_CONFIG)
        await integration.initialize()
        await integration.shutdown()

        assert integration.is_initialized is False


# ============================================================================
# Test Suite 5: Consciousness Orchestrator with LLM Tests
# ============================================================================

class TestConsciousnessOrchestratorWithLLM:
    """Test consciousness orchestrator with LLM integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_with_llm_initialization(self):
        """Test orchestrator initialization with LLM."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True
        )

        success = await orchestrator.initialize()

        assert success is True
        assert orchestrator.llm_integration is not None
        assert orchestrator.llm_integration.is_initialized is True

    @pytest.mark.asyncio
    async def test_orchestrator_without_llm_initialization(self):
        """Test orchestrator initialization without LLM."""
        orchestrator = ConsciousnessOrchestrator(enable_llm=False)

        success = await orchestrator.initialize()

        assert success is True
        assert orchestrator.llm_integration is None

    @pytest.mark.asyncio
    async def test_conscious_interaction_with_llm(self):
        """Test conscious interaction with LLM enabled."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True
        )
        await orchestrator.initialize()

        response = await orchestrator.process_conscious_interaction(
            "What is consciousness?"
        )

        assert response is not None
        assert len(response.response_text) > 0
        assert response.phi_during_response > 0

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_conscious_interaction_without_llm(self):
        """Test conscious interaction without LLM."""
        orchestrator = ConsciousnessOrchestrator(enable_llm=False)
        await orchestrator.initialize()

        response = await orchestrator.process_conscious_interaction(
            "What is consciousness?"
        )

        assert response is not None
        assert len(response.response_text) > 0
        assert response.phi_during_response > 0

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_qualia_enhancement_with_llm(self):
        """Test qualia enhancement with LLM."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True,
            enable_qualia_simulation=True
        )
        await orchestrator.initialize()

        response = await orchestrator.process_conscious_interaction(
            "Describe the feeling of experiencing beauty"
        )

        assert response.qualia_description is not None
        assert len(response.qualia_description) > 0

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_llm_stats_from_orchestrator(self):
        """Test getting LLM stats from orchestrator."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True
        )
        await orchestrator.initialize()

        # Generate some interactions
        await orchestrator.process_conscious_interaction("Test 1")
        await orchestrator.process_conscious_interaction("Test 2")

        stats = orchestrator.get_llm_stats()

        assert stats is not None
        assert stats['total_generations'] >= 2
        assert stats['backend'] == 'mock'

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_consciousness_metrics_with_llm(self):
        """Test consciousness metrics with LLM usage."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True
        )
        await orchestrator.initialize()

        # Generate interactions
        await orchestrator.process_conscious_interaction("Hello")
        await orchestrator.process_conscious_interaction("How are you?")

        metrics = orchestrator.get_consciousness_metrics()

        assert 'llm_usage' in metrics
        assert metrics['llm_usage']['total_generations'] >= 2

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_shutdown_with_llm(self):
        """Test orchestrator shutdown with LLM."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True
        )
        await orchestrator.initialize()
        await orchestrator.shutdown()

        # LLM integration should be shut down
        assert orchestrator.llm_integration.is_initialized is False


# ============================================================================
# Test Suite 6: Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_complete_consciousness_workflow_with_llm(self):
        """Test complete consciousness workflow with LLM."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True,
            enable_emotional_processing=True,
            enable_qualia_simulation=True
        )
        await orchestrator.initialize()

        # Process empathetic query
        response = await orchestrator.process_conscious_interaction(
            "I'm feeling uncertain about consciousness. Can you help?"
        )

        # Verify all consciousness components
        assert response.phi_during_response > 0
        assert response.consciousness_state is not None
        assert response.consciousness_state.empathy_level > 0.5
        assert response.qualia_description is not None
        assert response.metacognitive_note is not None
        assert len(response.response_text) > 0

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_interactions_with_llm(self):
        """Test multiple interactions maintaining state."""
        orchestrator = ConsciousnessOrchestrator(
            llm_config=MOCK_CONFIG,
            enable_llm=True
        )
        await orchestrator.initialize()

        interactions = [
            "What is consciousness?",
            "How do you experience it?",
            "Can you reflect on that?"
        ]

        responses = []
        for text in interactions:
            response = await orchestrator.process_conscious_interaction(text)
            responses.append(response)

        # Verify all responses generated
        assert len(responses) == 3
        for resp in responses:
            assert len(resp.response_text) > 0
            assert resp.phi_during_response > 0

        # Check LLM usage
        stats = orchestrator.get_llm_stats()
        assert stats['total_generations'] >= 3

        await orchestrator.shutdown()


# ============================================================================
# Run Tests
# ============================================================================

def run_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING LLM INTEGRATION TESTS")
    print("=" * 80 + "\n")

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
