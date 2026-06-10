"""
LLM Client for vLLM/OpenAI Compatible Servers.

Provides async/sync interfaces for code generation with retry logic,
multi-model support, and comprehensive error handling.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
from openai import OpenAI, AsyncOpenAI
from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError, NotFoundError


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    code: str
    raw_response: str
    tokens_used: int
    latency_ms: float
    model_name: str
    finish_reason: Optional[str] = None
    error: Optional[str] = None


class LLMClient:
    """
    OpenAI-compatible client for vLLM and OpenAI servers.

    Supports both synchronous and asynchronous generation with:
    - Exponential backoff retry (3 attempts)
    - Model configuration loading from YAML
    - Health check capabilities
    - Token usage tracking
    - Multiple providers: vLLM (local) and OpenAI (commercial API)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        timeout: float = 180.0,
        max_retries: int = 3,
        config_path: Optional[Path] = None,
        provider: str = "vllm",
    ):
        """
        Initialize LLM client.

        Args:
            base_url: Server base URL (ignored for OpenAI provider)
            api_key: API key (use "EMPTY" for local vLLM, or provide OpenAI key)
            model: Model name or identifier
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts with exponential backoff
            config_path: Path to models.yaml config file
            provider: Provider type - "vllm" for local server, "openai" for OpenAI API,
                      or "mindlogic" for Mindlogic AI Gateway (supports OpenAI/Anthropic/Google models)
        """
        self.provider = provider.lower()
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure provider-specific settings
        if self.provider == "mindlogic":
            # Mindlogic AI Gateway — unified endpoint for OpenAI/Anthropic/Google models
            self.base_url = "https://factchat-cloud.mindlogic.ai/v1/gateway"
            if api_key == "EMPTY" or not api_key:
                api_key = os.getenv("MINDLOGIC_API_KEY", "")
                if not api_key:
                    raise ValueError(
                        "Mindlogic API key required. Set MINDLOGIC_API_KEY environment variable "
                        "or pass api_key parameter."
                    )
            self.api_key = api_key
        elif self.provider == "openai":
            # OpenAI API configuration
            self.base_url = "https://api.openai.com/v1"
            # Try to get API key from parameter, environment, or raise error
            if api_key == "EMPTY" or not api_key:
                api_key = os.getenv("OPENAI_API_KEY", "")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                        "or pass api_key parameter."
                    )
            self.api_key = api_key
        else:
            # vLLM (local server) configuration
            self.base_url = base_url
            self.api_key = api_key

        # Load model configuration
        if config_path is None:
            # Auto-detect config path
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "model" / "models.yaml"

        self.model_config = self._load_model_config(config_path)

        # Cache for models that require completions API (non-chat models)
        self._non_chat_models: set = set()
        # Cache for models that require Responses API (not chat, not completions)
        # Pre-populate known Responses-only models to avoid slow fallback chain
        self._responses_only_models: set = set()
        _known_responses_models = {"gpt-5.2-pro", "o1", "o1-pro", "o3", "o3-pro"}
        resolved_name = self._resolve_model_name(model)
        if model in _known_responses_models or resolved_name in _known_responses_models:
            self._responses_only_models.add(resolved_name)

        # Models that do not support the temperature parameter
        self._no_temperature_models: set = set()
        _known_no_temp = {"claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5-20251101",
                          "gpt-5.2-pro", "o1", "o1-pro", "o3", "o3-pro"}
        if model in _known_no_temp or resolved_name in _known_no_temp:
            self._no_temperature_models.add(resolved_name)

        # Models whose chat template rejects a separate ``system`` role
        # (e.g. AutoTriton / Seed-Coder). For these, the system message is
        # folded into the user prompt instead of sent as its own role.
        _name_l = f"{model} {resolved_name}".lower()
        self._no_system_role = ("autotriton" in _name_l or "seed-coder" in _name_l)

        # Initialize OpenAI clients
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )

        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )

    def _load_model_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load model configuration from YAML.

        Args:
            config_path: Path to models.yaml

        Returns:
            Dictionary mapping model aliases to configurations
        """
        if not config_path.exists():
            return {}

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return data.get("models", {})

    def _resolve_model_name(self, model_id: str) -> str:
        """
        Resolve model alias to full HuggingFace name.

        Args:
            model_id: Model alias (e.g., "qwen2.5-coder-32b") or full name

        Returns:
            Full model name for vLLM server
        """
        # Check if it's an alias in config
        if model_id in self.model_config:
            return self.model_config[model_id].get("name", model_id)

        # Return as-is if not found (assume it's already a full name)
        return model_id

    def _get_model_params(self, model_id: str) -> Dict[str, Any]:
        """
        Get generation parameters for a specific model.

        Args:
            model_id: Model alias or full name

        Returns:
            Dictionary of model-specific parameters
        """
        if model_id in self.model_config:
            config = self.model_config[model_id]
            return {
                "max_tokens": config.get("max_model_len", 8192),
                "temperature": 0.3,  # Default, can be overridden
            }

        return {"max_tokens": 8192, "temperature": 0.3}

    @staticmethod
    def _is_responses_only_error(error: Exception) -> bool:
        """
        Check if the error indicates the model only supports the Responses API.

        Args:
            error: The API error to check

        Returns:
            True if the model needs the Responses API
        """
        error_msg = str(error).lower()
        return (
            "not supported" in error_msg
            and ("v1/completions" in error_msg or "completions" in error_msg)
        )

    @staticmethod
    def _is_non_chat_error(error: Exception) -> bool:
        """
        Check if the error indicates the model doesn't support chat completions.

        Args:
            error: The API error to check

        Returns:
            True if the error indicates a non-chat model
        """
        error_msg = str(error).lower()
        return (
            isinstance(error, (NotFoundError, APIError))
            and ("not a chat model" in error_msg or "chat/completions" in error_msg)
        )

    def _generate_completions(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]],
        system_message: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate using the text completions API (v1/completions) for non-chat models.

        Args:
            prompt: Input prompt
            model_name: Resolved model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            system_message: Optional system message to prepend

        Returns:
            GenerationResult with code and metadata
        """
        # Combine system message and prompt into a single text prompt
        full_prompt = ""
        if system_message:
            full_prompt += system_message + "\n\n"
        full_prompt += prompt

        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = self.client.completions.create(
                    model=model_name,
                    prompt=full_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )

                latency_ms = (time.time() - start_time) * 1000

                choice = response.choices[0]
                code = choice.text or ""
                finish_reason = choice.finish_reason

                tokens_used = 0
                if response.usage:
                    tokens_used = response.usage.total_tokens

                return GenerationResult(
                    code=code,
                    raw_response=code,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=model_name,
                    finish_reason=finish_reason,
                )

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    time.sleep(delay)
                    continue

                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Completions API failed after {self.max_retries} retries: {last_error}",
                )

            except (NotFoundError, APIError) as e:
                # Model doesn't support completions either → try Responses API
                if self._is_responses_only_error(e):
                    self._responses_only_models.add(model_name)
                    return self._generate_responses(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_message=system_message,
                    )
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Completions API error: {str(e)}",
                )

            except Exception as e:
                # Check if this is a responses-only model error
                if self._is_responses_only_error(e):
                    self._responses_only_models.add(model_name)
                    return self._generate_responses(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        system_message=system_message,
                    )
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Completions API error: {str(e)}",
                )

        return GenerationResult(
            code="",
            raw_response="",
            tokens_used=0,
            latency_ms=0.0,
            model_name=model_name,
            error=f"Completions API failed after {self.max_retries} retries",
        )

    def _generate_responses(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate using the Responses API (v1/responses) for models like gpt-5.2-pro.

        Args:
            prompt: Input prompt
            model_name: Resolved model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_message: Optional system instructions

        Returns:
            GenerationResult with code and metadata
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                kwargs = {
                    "model": model_name,
                    "input": prompt,
                    "max_output_tokens": max_tokens,
                }
                if system_message:
                    kwargs["instructions"] = system_message
                # Reasoning models (gpt-5.2-pro, o1, o3) don't support temperature
                # Skip temperature for these models to avoid API errors

                response = self.client.responses.create(**kwargs)

                latency_ms = (time.time() - start_time) * 1000

                code = response.output_text or ""

                tokens_used = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = getattr(response.usage, 'total_tokens', 0)

                return GenerationResult(
                    code=code,
                    raw_response=code,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=model_name,
                    finish_reason="stop",
                )

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    time.sleep(delay)
                    continue

                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Responses API failed after {self.max_retries} retries: {last_error}",
                )

            except Exception as e:
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Responses API error: {str(e)}",
                )

        return GenerationResult(
            code="",
            raw_response="",
            tokens_used=0,
            latency_ms=0.0,
            model_name=model_name,
            error=f"Responses API failed after {self.max_retries} retries",
        )

    async def _agenerate_responses(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        system_message: Optional[str] = None,
    ) -> GenerationResult:
        """
        Async generate using the Responses API for models like gpt-5.2-pro.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                kwargs = {
                    "model": model_name,
                    "input": prompt,
                    "max_output_tokens": max_tokens,
                }
                if system_message:
                    kwargs["instructions"] = system_message
                # Skip temperature for reasoning models

                response = await self.async_client.responses.create(**kwargs)

                latency_ms = (time.time() - start_time) * 1000

                code = response.output_text or ""

                tokens_used = 0
                if hasattr(response, 'usage') and response.usage:
                    tokens_used = getattr(response.usage, 'total_tokens', 0)

                return GenerationResult(
                    code=code,
                    raw_response=code,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=model_name,
                    finish_reason="stop",
                )

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    await asyncio.sleep(delay)
                    continue

                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Responses API failed after {self.max_retries} retries: {last_error}",
                )

            except Exception as e:
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Responses API error: {str(e)}",
                )

        return GenerationResult(
            code="",
            raw_response="",
            tokens_used=0,
            latency_ms=0.0,
            model_name=model_name,
            error=f"Responses API failed after {self.max_retries} retries",
        )

    async def _agenerate_completions(
        self,
        prompt: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        stop: Optional[List[str]],
        system_message: Optional[str] = None,
    ) -> GenerationResult:
        """
        Async generate using the text completions API (v1/completions) for non-chat models.

        Args:
            prompt: Input prompt
            model_name: Resolved model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            system_message: Optional system message to prepend

        Returns:
            GenerationResult with code and metadata
        """
        full_prompt = ""
        if system_message:
            full_prompt += system_message + "\n\n"
        full_prompt += prompt

        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                response = await self.async_client.completions.create(
                    model=model_name,
                    prompt=full_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )

                latency_ms = (time.time() - start_time) * 1000

                choice = response.choices[0]
                code = choice.text or ""
                finish_reason = choice.finish_reason

                tokens_used = 0
                if response.usage:
                    tokens_used = response.usage.total_tokens

                return GenerationResult(
                    code=code,
                    raw_response=code,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=model_name,
                    finish_reason=finish_reason,
                )

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    await asyncio.sleep(delay)
                    continue

                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Completions API failed after {self.max_retries} retries: {last_error}",
                )

            except Exception as e:
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Completions API error: {str(e)}",
                )

        return GenerationResult(
            code="",
            raw_response="",
            tokens_used=0,
            latency_ms=0.0,
            model_name=model_name,
            error=f"Completions API failed after {self.max_retries} retries",
        )

    def _exponential_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        return min(2 ** attempt, 30)  # Max 30 seconds

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate code synchronously with retry logic.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            model: Override default model
            system_message: Optional system role message for guiding LLM behavior

        Returns:
            GenerationResult with code and metadata
        """
        model_id = model or self.model
        model_name = self._resolve_model_name(model_id)
        model_params = self._get_model_params(model_id)

        # Models that reject a separate system role: fold it into the prompt.
        if getattr(self, "_no_system_role", False) and system_message:
            prompt = system_message + "\n\n" + prompt
            system_message = None

        # Override defaults with provided values
        if temperature is None:
            temperature = model_params.get("temperature", 0.3)
        if max_tokens is None:
            max_tokens = model_params.get("max_tokens", 8192)

        # If model is known to need Responses API, go directly
        if model_name in self._responses_only_models:
            return self._generate_responses(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_message,
            )

        # If model is known to be non-chat, go directly to completions API
        if model_name in self._non_chat_models:
            return self._generate_completions(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                system_message=system_message,
            )

        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Build messages with optional system message
                messages: List[Dict[str, str]] = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                # OpenAI newer models require max_completion_tokens instead of max_tokens
                token_param = (
                    {"max_completion_tokens": max_tokens}
                    if self.provider == "openai"
                    else {"max_tokens": max_tokens}
                )

                # Some models (e.g. claude-opus-4-7) don't support temperature
                sampling_params = {}
                if model_name not in self._no_temperature_models:
                    sampling_params["temperature"] = temperature
                    sampling_params["top_p"] = top_p

                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stop=stop,
                    **sampling_params,
                    **token_param,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract response
                choice = response.choices[0]
                code = choice.message.content or ""
                finish_reason = choice.finish_reason

                # Extract token usage
                tokens_used = 0
                if response.usage:
                    tokens_used = response.usage.total_tokens

                return GenerationResult(
                    code=code,
                    raw_response=code,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=model_name,
                    finish_reason=finish_reason,
                )

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    time.sleep(delay)
                    continue

                # Final attempt failed
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Failed after {self.max_retries} retries: {last_error}",
                )

            except (NotFoundError, APIError) as e:
                error_msg = str(e).lower()

                # Check if temperature is not supported → retry without it
                if "temperature" in error_msg and ("deprecated" in error_msg or "not supported" in error_msg):
                    self._no_temperature_models.add(model_name)
                    continue  # Retry — next attempt will skip temperature

                # Check if this is a non-chat model error → fallback to completions API
                if self._is_non_chat_error(e):
                    self._non_chat_models.add(model_name)
                    return self._generate_completions(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop,
                        system_message=system_message,
                    )

                # Non-retryable API error
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"API error: {str(e)}",
                )

            except Exception as e:
                # Unexpected error
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Unexpected error: {str(e)}",
                )

        # Should not reach here
        return GenerationResult(
            code="",
            raw_response="",
            tokens_used=0,
            latency_ms=0.0,
            model_name=model_name,
            error=f"Failed after {self.max_retries} retries",
        )

    async def agenerate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate code asynchronously with retry logic.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            model: Override default model
            system_message: Optional system role message for guiding LLM behavior

        Returns:
            GenerationResult with code and metadata
        """
        model_id = model or self.model
        model_name = self._resolve_model_name(model_id)
        model_params = self._get_model_params(model_id)

        # Models that reject a separate system role: fold it into the prompt.
        if getattr(self, "_no_system_role", False) and system_message:
            prompt = system_message + "\n\n" + prompt
            system_message = None

        # Override defaults with provided values
        if temperature is None:
            temperature = model_params.get("temperature", 0.3)
        if max_tokens is None:
            max_tokens = model_params.get("max_tokens", 8192)

        # If model is known to need Responses API, go directly
        if model_name in self._responses_only_models:
            return await self._agenerate_responses(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                system_message=system_message,
            )

        # If model is known to be non-chat, go directly to completions API
        if model_name in self._non_chat_models:
            return await self._agenerate_completions(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                system_message=system_message,
            )

        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()

                # Build messages with optional system message
                messages: List[Dict[str, str]] = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                # OpenAI newer models require max_completion_tokens instead of max_tokens
                token_param = (
                    {"max_completion_tokens": max_tokens}
                    if self.provider == "openai"
                    else {"max_tokens": max_tokens}
                )

                response = await self.async_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **token_param,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Extract response
                choice = response.choices[0]
                code = choice.message.content or ""
                finish_reason = choice.finish_reason

                # Extract token usage
                tokens_used = 0
                if response.usage:
                    tokens_used = response.usage.total_tokens

                return GenerationResult(
                    code=code,
                    raw_response=code,
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model_name=model_name,
                    finish_reason=finish_reason,
                )

            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_error = str(e)

                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    await asyncio.sleep(delay)
                    continue

                # Final attempt failed
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Failed after {self.max_retries} retries: {last_error}",
                )

            except (NotFoundError, APIError) as e:
                # Check if this is a non-chat model error → fallback to completions API
                if self._is_non_chat_error(e):
                    self._non_chat_models.add(model_name)
                    return await self._agenerate_completions(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stop=stop,
                        system_message=system_message,
                    )

                # Non-retryable API error
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"API error: {str(e)}",
                )

            except Exception as e:
                # Unexpected error
                return GenerationResult(
                    code="",
                    raw_response="",
                    tokens_used=0,
                    latency_ms=0.0,
                    model_name=model_name,
                    error=f"Unexpected error: {str(e)}",
                )

        # Should not reach here
        return GenerationResult(
            code="",
            raw_response="",
            tokens_used=0,
            latency_ms=0.0,
            model_name=model_name,
            error=f"Failed after {self.max_retries} retries",
        )

    def health_check(self) -> bool:
        """
        Check if the vLLM server is healthy.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            # Try to list models as a simple health check
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    async def ahealth_check(self) -> bool:
        """
        Async health check for vLLM server.

        Returns:
            True if server is responding, False otherwise
        """
        try:
            # Try to list models as a simple health check
            models = await self.async_client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def list_available_models(self) -> List[str]:
        """
        List models available on the server.

        Returns:
            List of model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            return []

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration info for a model.

        Args:
            model_id: Model alias or full name

        Returns:
            Model configuration dictionary or None
        """
        return self.model_config.get(model_id)
