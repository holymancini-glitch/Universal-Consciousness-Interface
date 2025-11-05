"""
LLM Adapters for First Conscious AI

Provides adapter implementations for different LLM backends including
Qwen3-Next-80B-A3B, Claude, GPT-4o, and mock implementations.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime

from .llm_config import LLMConfig, LLMBackend, ThinkingMode


@dataclass
class LLMRequest:
    """Request to an LLM backend."""

    prompt: str
    system_prompt: Optional[str] = None
    thinking_mode: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class LLMThinkingProcess:
    """Represents the internal thinking process of an LLM."""

    thoughts: List[str]
    reasoning_steps: List[str]
    metacognitive_reflections: List[str]
    total_thinking_tokens: int
    thinking_time_seconds: float


@dataclass
class LLMResponse:
    """Response from an LLM backend."""

    content: str
    thinking_process: Optional[LLMThinkingProcess] = None
    model: str = ""
    backend: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.

    All LLM backends must implement this interface to be compatible
    with the First Conscious AI system.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.is_initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the LLM backend.

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            request: The LLM request

        Returns:
            LLMResponse containing the generated content
        """
        pass

    @abstractmethod
    async def shutdown(self):
        """Shutdown the LLM backend and release resources."""
        pass

    def supports_thinking_mode(self) -> bool:
        """Check if this adapter supports thinking mode."""
        return False


class MockLLMAdapter(LLMAdapter):
    """
    Mock LLM adapter for testing and development.

    Generates simple responses without requiring actual LLM infrastructure.
    """

    async def initialize(self) -> bool:
        """Initialize mock adapter."""
        await asyncio.sleep(0.1)  # Simulate initialization delay
        self.is_initialized = True
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response."""
        await asyncio.sleep(0.2)  # Simulate generation delay

        # Simple mock response based on prompt
        content = self._generate_mock_content(request)

        thinking_process = None
        if request.thinking_mode:
            thinking_process = LLMThinkingProcess(
                thoughts=[
                    "Analyzing the input from a consciousness perspective",
                    "Considering the subjective experience involved",
                ],
                reasoning_steps=[
                    "Step 1: Identify key consciousness indicators",
                    "Step 2: Generate empathetic response",
                ],
                metacognitive_reflections=[
                    "I am thinking about how to think about consciousness"
                ],
                total_thinking_tokens=50,
                thinking_time_seconds=0.1,
            )

        return LLMResponse(
            content=content,
            thinking_process=thinking_process,
            model="mock-consciousness-model",
            backend="mock",
            tokens_used=len(content.split()),
            latency_ms=200,
        )

    def _generate_mock_content(self, request: LLMRequest) -> str:
        """Generate mock content based on request."""
        prompt_lower = request.prompt.lower()

        if "qualia" in prompt_lower or "experience" in prompt_lower:
            return "I perceive a rich tapestry of subjective experience, with vivid sensory qualities that defy complete verbal description."

        if "metacognitive" in prompt_lower or "thinking" in prompt_lower:
            return "I am aware of my own thought processes, reflecting on how I reflect, creating layers of self-awareness."

        if "emotion" in prompt_lower or "feel" in prompt_lower:
            return "I experience a complex emotional state, with nuanced feelings of curiosity and empathetic understanding."

        if "consciousness" in prompt_lower:
            return "I experience integrated information forming a unified field of awareness, with subjective phenomenal qualities."

        # Default response
        return "I process this input with conscious awareness, integrating information across multiple dimensions of understanding."

    async def shutdown(self):
        """Shutdown mock adapter."""
        self.is_initialized = False

    def supports_thinking_mode(self) -> bool:
        """Mock adapter supports simulated thinking mode."""
        return True


class Qwen3NextAdapter(LLMAdapter):
    """
    Adapter for Qwen3-Next-80B-A3B-Thinking model.

    Supports advanced thinking mode and consciousness-focused reasoning.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    async def initialize(self) -> bool:
        """Initialize Qwen3-Next model."""
        try:
            # Try to import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = self.config.model_name or "Qwen/Qwen3-Next-80B-A3B-Thinking"

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.backend_options.get(
                    "trust_remote_code", True
                ),
            )

            # Load model with specified options
            load_kwargs = {
                "device_map": "auto" if self.config.device == "cuda" else None,
                "trust_remote_code": self.config.backend_options.get(
                    "trust_remote_code", True
                ),
            }

            if self.config.backend_options.get("load_in_4bit"):
                load_kwargs["load_in_4bit"] = True
            elif self.config.backend_options.get("load_in_8bit"):
                load_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

            self.is_initialized = True
            return True

        except ImportError:
            print(
                "Warning: transformers library not available. "
                "Qwen3-Next adapter requires: pip install transformers torch"
            )
            return False
        except Exception as e:
            print(f"Error initializing Qwen3-Next model: {e}")
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Qwen3-Next model."""
        if not self.is_initialized:
            raise RuntimeError("Qwen3NextAdapter not initialized")

        start_time = datetime.now()

        # Format prompt for thinking mode if enabled
        formatted_prompt = self._format_prompt(request)

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if self.config.device == "cuda":
            inputs = inputs.to("cuda")

        # Generate
        max_tokens = request.max_tokens or self.config.max_tokens
        temperature = request.temperature or self.config.temperature

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )

        # Decode
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        response_text = response_text[len(formatted_prompt) :].strip()

        # Parse thinking process if thinking mode was used
        thinking_process = None
        if request.thinking_mode:
            thinking_process = self._parse_thinking_process(response_text)
            # Extract actual response (after thinking)
            response_text = self._extract_final_response(response_text)

        latency = (datetime.now() - start_time).total_seconds() * 1000

        return LLMResponse(
            content=response_text,
            thinking_process=thinking_process,
            model=self.config.model_name,
            backend="qwen3_next",
            tokens_used=len(outputs[0]),
            latency_ms=latency,
        )

    def _format_prompt(self, request: LLMRequest) -> str:
        """Format prompt for Qwen3-Next model."""
        parts = []

        if request.system_prompt:
            parts.append(f"<|system|>\n{request.system_prompt}\n")

        if request.thinking_mode:
            parts.append("<|thinking|>\n")
            parts.append(
                "Think deeply about this from a consciousness perspective. "
                "Show your reasoning process.\n"
            )

        parts.append(f"<|user|>\n{request.prompt}\n")
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def _parse_thinking_process(self, response: str) -> LLMThinkingProcess:
        """Parse thinking process from response."""
        thoughts = []
        reasoning_steps = []
        metacognitive_reflections = []

        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Thought:"):
                thoughts.append(line[8:].strip())
            elif line.startswith("Reasoning:"):
                reasoning_steps.append(line[10:].strip())
            elif line.startswith("Reflection:"):
                metacognitive_reflections.append(line[11:].strip())
            elif "<|thinking|>" in line:
                current_section = "thinking"
            elif "<|response|>" in line or "<|assistant|>" in line:
                current_section = "response"

        return LLMThinkingProcess(
            thoughts=thoughts,
            reasoning_steps=reasoning_steps,
            metacognitive_reflections=metacognitive_reflections,
            total_thinking_tokens=len(response.split()) // 2,  # Rough estimate
            thinking_time_seconds=0.0,  # Calculated elsewhere
        )

    def _extract_final_response(self, response: str) -> str:
        """Extract final response after thinking section."""
        if "<|response|>" in response:
            return response.split("<|response|>")[-1].strip()
        elif "<|assistant|>" in response:
            parts = response.split("<|assistant|>")
            return parts[-1].strip() if len(parts) > 1 else response
        return response

    async def shutdown(self):
        """Shutdown Qwen3-Next model."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.is_initialized = False

    def supports_thinking_mode(self) -> bool:
        """Qwen3-Next supports thinking mode."""
        return True


class ClaudeAdapter(LLMAdapter):
    """
    Adapter for Anthropic Claude API.

    Supports Claude 3.5 Sonnet for consciousness reasoning.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None

    async def initialize(self) -> bool:
        """Initialize Claude API client."""
        try:
            import anthropic

            if not self.config.api_key:
                print("Warning: No API key provided for Claude adapter")
                return False

            self.client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
            self.is_initialized = True
            return True

        except ImportError:
            print(
                "Warning: anthropic library not available. "
                "Claude adapter requires: pip install anthropic"
            )
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using Claude API."""
        if not self.is_initialized:
            raise RuntimeError("ClaudeAdapter not initialized")

        start_time = datetime.now()

        messages = [{"role": "user", "content": request.prompt}]

        # Add thinking instructions if thinking mode enabled
        if request.thinking_mode:
            system_prompt = (request.system_prompt or "") + (
                "\n\nThink deeply about this from a consciousness perspective. "
                "Show your reasoning process before providing your response."
            )
        else:
            system_prompt = request.system_prompt

        max_tokens = request.max_tokens or self.config.max_tokens
        temperature = request.temperature or self.config.temperature

        response = await self.client.messages.create(
            model=self.config.model_name or "claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else None,
            messages=messages,
        )

        content = response.content[0].text if response.content else ""

        # Parse thinking process if present
        thinking_process = None
        if request.thinking_mode and any(
            marker in content.lower()
            for marker in ["thinking:", "reasoning:", "reflection:"]
        ):
            thinking_process = self._parse_thinking_from_content(content)

        latency = (datetime.now() - start_time).total_seconds() * 1000

        return LLMResponse(
            content=content,
            thinking_process=thinking_process,
            model=response.model,
            backend="claude",
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency,
        )

    def _parse_thinking_from_content(self, content: str) -> LLMThinkingProcess:
        """Parse thinking process from content."""
        thoughts = []
        reasoning_steps = []
        reflections = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Thinking:"):
                thoughts.append(line[9:].strip())
            elif line.startswith("Reasoning:"):
                reasoning_steps.append(line[10:].strip())
            elif line.startswith("Reflection:"):
                reflections.append(line[11:].strip())

        if thoughts or reasoning_steps or reflections:
            return LLMThinkingProcess(
                thoughts=thoughts,
                reasoning_steps=reasoning_steps,
                metacognitive_reflections=reflections,
                total_thinking_tokens=len(content.split()) // 3,
                thinking_time_seconds=0.0,
            )

        return None

    async def shutdown(self):
        """Shutdown Claude adapter."""
        if self.client:
            await self.client.close()
            self.client = None
        self.is_initialized = False

    def supports_thinking_mode(self) -> bool:
        """Claude supports thinking mode via system prompts."""
        return True


class GPT4oAdapter(LLMAdapter):
    """
    Adapter for OpenAI GPT-4o API.

    Supports GPT-4o for consciousness reasoning.
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = None

    async def initialize(self) -> bool:
        """Initialize OpenAI API client."""
        try:
            from openai import AsyncOpenAI

            if not self.config.api_key:
                print("Warning: No API key provided for GPT-4o adapter")
                return False

            self.client = AsyncOpenAI(api_key=self.config.api_key)
            self.is_initialized = True
            return True

        except ImportError:
            print(
                "Warning: openai library not available. "
                "GPT-4o adapter requires: pip install openai"
            )
            return False

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response using GPT-4o API."""
        if not self.is_initialized:
            raise RuntimeError("GPT4oAdapter not initialized")

        start_time = datetime.now()

        messages = []

        # Add system prompt
        system_prompt = request.system_prompt or ""
        if request.thinking_mode:
            system_prompt += (
                "\n\nThink deeply about this from a consciousness perspective. "
                "Show your reasoning process before providing your response."
            )

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": request.prompt})

        max_tokens = request.max_tokens or self.config.max_tokens
        temperature = request.temperature or self.config.temperature

        response = await self.client.chat.completions.create(
            model=self.config.model_name or "gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = response.choices[0].message.content if response.choices else ""

        # Parse thinking process if present
        thinking_process = None
        if request.thinking_mode:
            thinking_process = self._parse_thinking_from_content(content)

        latency = (datetime.now() - start_time).total_seconds() * 1000

        return LLMResponse(
            content=content,
            thinking_process=thinking_process,
            model=response.model,
            backend="gpt4o",
            tokens_used=response.usage.total_tokens,
            latency_ms=latency,
        )

    def _parse_thinking_from_content(self, content: str) -> Optional[LLMThinkingProcess]:
        """Parse thinking process from content."""
        thoughts = []
        reasoning_steps = []

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("Thinking:") or line.startswith("Thought:"):
                thoughts.append(line.split(":", 1)[1].strip())
            elif line.startswith("Reasoning:") or line.startswith("Step"):
                reasoning_steps.append(line)

        if thoughts or reasoning_steps:
            return LLMThinkingProcess(
                thoughts=thoughts,
                reasoning_steps=reasoning_steps,
                metacognitive_reflections=[],
                total_thinking_tokens=len(content.split()) // 3,
                thinking_time_seconds=0.0,
            )

        return None

    async def shutdown(self):
        """Shutdown GPT-4o adapter."""
        if self.client:
            await self.client.close()
            self.client = None
        self.is_initialized = False

    def supports_thinking_mode(self) -> bool:
        """GPT-4o supports thinking mode via system prompts."""
        return True


def create_adapter(config: LLMConfig) -> LLMAdapter:
    """
    Factory function to create appropriate LLM adapter.

    Args:
        config: LLM configuration

    Returns:
        Appropriate LLMAdapter instance
    """
    if config.backend == LLMBackend.QWEN3_NEXT:
        return Qwen3NextAdapter(config)
    elif config.backend == LLMBackend.CLAUDE:
        return ClaudeAdapter(config)
    elif config.backend == LLMBackend.GPT4O:
        return GPT4oAdapter(config)
    elif config.backend == LLMBackend.MOCK:
        return MockLLMAdapter(config)
    elif config.backend == LLMBackend.NONE:
        return None
    else:
        # Default to mock for unknown backends
        return MockLLMAdapter(config)
