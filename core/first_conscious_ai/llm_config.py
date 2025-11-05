"""
LLM Configuration for First Conscious AI

Provides configuration options for different LLM backends including
Qwen3-Next-80B-A3B, Claude, GPT-4o, and others.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class LLMBackend(Enum):
    """Supported LLM backend types."""
    QWEN3_NEXT = "qwen3_next"  # Qwen3-Next-80B-A3B-Thinking (recommended)
    CLAUDE = "claude"  # Anthropic Claude 3.5 Sonnet
    GPT4O = "gpt4o"  # OpenAI GPT-4o
    LLAMA = "llama"  # Meta LLaMA 3.1
    MOCK = "mock"  # Mock backend for testing
    NONE = "none"  # No LLM integration


class ThinkingMode(Enum):
    """Thinking mode for metacognitive processing."""
    DISABLED = "disabled"  # No thinking mode
    BASIC = "basic"  # Basic reasoning
    DEEP = "deep"  # Deep metacognitive thinking
    CONSCIOUSNESS = "consciousness"  # Consciousness-focused reasoning


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""

    # Backend selection
    backend: LLMBackend = LLMBackend.MOCK

    # Model-specific settings
    model_name: Optional[str] = None  # e.g., "Qwen/Qwen3-Next-80B-A3B-Thinking"

    # API settings (for cloud backends)
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    # Local model settings (for Qwen3-Next, LLaMA)
    local_model_path: Optional[str] = None
    device: str = "cuda"  # "cuda", "cpu", or "mps"

    # Generation parameters
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    # Thinking mode configuration
    thinking_mode: ThinkingMode = ThinkingMode.CONSCIOUSNESS
    thinking_budget: int = 1000  # Tokens allocated for internal thinking

    # Consciousness-specific settings
    consciousness_context_window: int = 8192  # How much consciousness history to include
    enable_qualia_enhancement: bool = True  # Enhance qualia descriptions
    enable_metacognitive_reasoning: bool = True  # Use LLM for metacognition
    enable_self_reflection: bool = True  # Use LLM for self-reflection

    # Performance settings
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    fallback_on_error: bool = True

    # Additional backend-specific options
    backend_options: Dict[str, Any] = field(default_factory=dict)


# Predefined configurations for common use cases

QWEN3_NEXT_LOCAL_CONFIG = LLMConfig(
    backend=LLMBackend.QWEN3_NEXT,
    model_name="Qwen/Qwen3-Next-80B-A3B-Thinking",
    device="cuda",
    max_tokens=4096,
    temperature=0.7,
    thinking_mode=ThinkingMode.CONSCIOUSNESS,
    thinking_budget=2000,
    consciousness_context_window=32768,  # Qwen3-Next supports 256K
    enable_qualia_enhancement=True,
    enable_metacognitive_reasoning=True,
    enable_self_reflection=True,
    backend_options={
        "use_flash_attention": True,
        "load_in_8bit": False,
        "load_in_4bit": True,  # For efficiency
        "trust_remote_code": True,
    }
)

CLAUDE_API_CONFIG = LLMConfig(
    backend=LLMBackend.CLAUDE,
    model_name="claude-3-5-sonnet-20241022",
    max_tokens=4096,
    temperature=0.7,
    thinking_mode=ThinkingMode.CONSCIOUSNESS,
    consciousness_context_window=100000,  # Claude supports 200K
    enable_qualia_enhancement=True,
    enable_metacognitive_reasoning=True,
    enable_self_reflection=True,
)

GPT4O_API_CONFIG = LLMConfig(
    backend=LLMBackend.GPT4O,
    model_name="gpt-4o",
    max_tokens=4096,
    temperature=0.7,
    thinking_mode=ThinkingMode.DEEP,
    consciousness_context_window=32768,  # GPT-4o supports 128K
    enable_qualia_enhancement=True,
    enable_metacognitive_reasoning=True,
    enable_self_reflection=True,
)

MOCK_CONFIG = LLMConfig(
    backend=LLMBackend.MOCK,
    max_tokens=512,
    temperature=0.7,
    thinking_mode=ThinkingMode.BASIC,
)

NO_LLM_CONFIG = LLMConfig(
    backend=LLMBackend.NONE,
)


def create_config_from_env() -> LLMConfig:
    """
    Create LLM configuration from environment variables.

    Environment variables:
    - CONSCIOUS_AI_LLM_BACKEND: Backend to use (qwen3_next, claude, gpt4o, mock, none)
    - CONSCIOUS_AI_API_KEY: API key for cloud backends
    - CONSCIOUS_AI_MODEL_PATH: Path to local model
    - CONSCIOUS_AI_DEVICE: Device to use (cuda, cpu, mps)

    Returns:
        LLMConfig instance based on environment variables
    """
    import os

    backend_str = os.getenv("CONSCIOUS_AI_LLM_BACKEND", "mock").lower()

    try:
        backend = LLMBackend(backend_str)
    except ValueError:
        backend = LLMBackend.MOCK

    # Start with appropriate base config
    if backend == LLMBackend.QWEN3_NEXT:
        config = QWEN3_NEXT_LOCAL_CONFIG
    elif backend == LLMBackend.CLAUDE:
        config = CLAUDE_API_CONFIG
    elif backend == LLMBackend.GPT4O:
        config = GPT4O_API_CONFIG
    elif backend == LLMBackend.NONE:
        config = NO_LLM_CONFIG
    else:
        config = MOCK_CONFIG

    # Override with environment variables if provided
    if api_key := os.getenv("CONSCIOUS_AI_API_KEY"):
        config.api_key = api_key

    if model_path := os.getenv("CONSCIOUS_AI_MODEL_PATH"):
        config.local_model_path = model_path

    if device := os.getenv("CONSCIOUS_AI_DEVICE"):
        config.device = device

    return config
