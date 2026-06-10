"""Initial Triton Kernel Generator.

Generates Triton GPU kernels from PyTorch reference code using an LLM,
with GPU-specific optimization hints and structured retry logic.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from llm_kernel.client.llm_client import LLMClient, GenerationResult
from llm_kernel.client.prompt_builder import PromptBuilder
from llm_kernel.client.response_parser import ResponseParser, ParseResult
from llm_kernel.knowledge.gpu_specs import GPUSpec
from llm_kernel.knowledge.triton_patterns import (
    get_pattern_for_operation,
    format_best_practices,
)
from llm_kernel.knowledge.blackwell_hints import (
    get_optimization_hints,
    format_general_hints,
)

logger = logging.getLogger(__name__)

_MAX_PARSE_RETRIES = 2

_FORMAT_RETRY_SUFFIX = """

IMPORTANT: Your previous response could not be parsed. You MUST return ONLY valid Python code
wrapped in a ```python code block. The code MUST contain:
1. A `class Model(nn.Module)` with `__init__` and `forward` methods
2. At least one `@triton.jit` decorated kernel function
3. A `def get_inputs()` function returning sample input tensors
4. A `def get_init_inputs()` function (return empty tuple if no init args needed)

Do NOT include any explanation text outside the code block.
"""


@dataclass
class GenerationOutput:
    """Result of a kernel generation attempt."""

    code: Optional[str]
    is_valid: bool
    parse_errors: list[str] = field(default_factory=list)
    attempts: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    model_name: str = ""


class KernelGenerator:
    """Generates Triton kernels from PyTorch reference code via LLM.

    Orchestrates prompt building with GPU specs and Triton patterns,
    LLM invocation, response parsing, and retry on parse failure.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        response_parser: ResponseParser,
        gpu_spec: GPUSpec,
    ):
        """Initialize the kernel generator.

        Args:
            llm_client: Client for LLM inference.
            prompt_builder: Jinja2-based prompt constructor.
            response_parser: Extracts and validates Triton code from LLM output.
            gpu_spec: Target GPU specification for optimization hints.
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.gpu_spec = gpu_spec

        # Push GPU spec into the prompt builder for template rendering
        self.prompt_builder.set_gpu_spec(self._gpu_spec_dict())

        # Build and cache the system prompt
        self.system_prompt = self.prompt_builder.build_system_prompt()

    def _gpu_spec_dict(self) -> dict:
        """Convert GPUSpec dataclass to dict for prompt templates."""
        return {
            "name": self.gpu_spec.name,
            "arch": self.gpu_spec.arch,
            "vram_gb": self.gpu_spec.vram_gb,
            "memory_bandwidth_gbs": self.gpu_spec.memory_bandwidth_gbs,
            "fp16_tflops": self.gpu_spec.fp16_tflops,
            "cuda_cores": self.gpu_spec.cuda_cores,
            "tensor_cores": self.gpu_spec.tensor_cores,
        }

    def _build_enriched_operation_desc(
        self, operation_desc: str, input_shapes: Optional[dict] = None
    ) -> str:
        """Build an enriched operation description with patterns and hints.

        Args:
            operation_desc: User-provided operation description.
            input_shapes: Optional mapping of input names to shape tuples.

        Returns:
            Enriched description string.
        """
        parts: list[str] = []
        if operation_desc:
            parts.append(operation_desc)

        if input_shapes:
            shape_lines = ", ".join(
                f"{name}: {shape}" for name, shape in input_shapes.items()
            )
            parts.append(f"Input shapes: {shape_lines}")

        # Try to detect operation type from description and inject patterns
        op_keywords = {
            "matmul": ["matmul", "matrix multiply", "mm", "gemm", "linear"],
            "softmax": ["softmax"],
            "attention": ["attention", "self-attention", "cross-attention"],
            "reduction": ["reduce", "sum", "mean", "max", "min", "norm"],
            "elementwise": ["elementwise", "element-wise", "relu", "gelu", "silu", "activation"],
            "normalization": ["layernorm", "layer_norm", "batchnorm", "batch_norm", "groupnorm"],
            "convolution": ["conv", "convolution"],
        }

        detected_op: Optional[str] = None
        desc_lower = operation_desc.lower()
        for op_type, keywords in op_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                detected_op = op_type
                break

        if detected_op:
            pattern = get_pattern_for_operation(detected_op)
            if pattern:
                parts.append(f"\nReference Triton pattern for {detected_op}:\n```\n{pattern}\n```")

            hints = get_optimization_hints(detected_op)
            if hints:
                hint_text = "\n".join(f"- {h}" for h in hints[:5])
                parts.append(f"\nOptimization hints ({detected_op}):\n{hint_text}")

        # Always add general best practices (abbreviated)
        parts.append(f"\n{format_best_practices()}")
        parts.append(f"\n{format_general_hints()}")

        return "\n\n".join(parts)

    def generate(
        self,
        pytorch_code: str,
        operation_desc: str = "",
        input_shapes: Optional[dict] = None,
        skill_memory: str = "",
    ) -> GenerationOutput:
        """Generate a Triton kernel from PyTorch reference code.

        Args:
            pytorch_code: Reference PyTorch implementation.
            operation_desc: Natural-language description of the operation.
            input_shapes: Optional mapping of input name to shape tuple.

        Returns:
            GenerationOutput with generated code, validity status, and metrics.
        """
        enriched_desc = self._build_enriched_operation_desc(operation_desc, input_shapes)

        total_tokens = 0
        total_latency = 0.0
        all_errors: list[str] = []

        for attempt in range(1, _MAX_PARSE_RETRIES + 2):  # 1 initial + up to 2 retries
            logger.info("Generation attempt %d/%d", attempt, _MAX_PARSE_RETRIES + 1)

            # On retry, append explicit format instruction
            if attempt == 1:
                prompt = self.prompt_builder.build_generation_prompt(
                    pytorch_code=pytorch_code,
                    operation_desc=enriched_desc,
                    skill_memory=skill_memory,
                )
            else:
                prompt = self.prompt_builder.build_generation_prompt(
                    pytorch_code=pytorch_code,
                    operation_desc=enriched_desc + _FORMAT_RETRY_SUFFIX,
                    skill_memory=skill_memory,
                )

            gen_result: GenerationResult = self.llm_client.generate(
                prompt, system_message=self.system_prompt
            )
            total_tokens += gen_result.tokens_used
            total_latency += gen_result.latency_ms

            # Check LLM-level error
            if gen_result.error:
                logger.error("LLM error on attempt %d: %s", attempt, gen_result.error)
                all_errors.append(f"LLM error: {gen_result.error}")
                continue

            # Parse the response
            parse_result: ParseResult = self.response_parser.parse(
                gen_result.code, original_pytorch_code=pytorch_code
            )

            if parse_result.is_valid:
                # Syntax check
                syntax_errors = self.response_parser.validate_syntax(parse_result.code)
                if syntax_errors:
                    logger.warning("Syntax errors on attempt %d: %s", attempt, syntax_errors)
                    all_errors.extend(syntax_errors)
                    continue

                logger.info("Valid kernel generated on attempt %d", attempt)
                return GenerationOutput(
                    code=parse_result.code,
                    is_valid=True,
                    parse_errors=[],
                    attempts=attempt,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency,
                    model_name=gen_result.model_name,
                )

            logger.warning(
                "Parse failed on attempt %d: %s", attempt, parse_result.parse_errors
            )
            all_errors.extend(parse_result.parse_errors)

        # All attempts exhausted
        logger.error("All generation attempts failed. Errors: %s", all_errors)
        return GenerationOutput(
            code=None,
            is_valid=False,
            parse_errors=all_errors,
            attempts=_MAX_PARSE_RETRIES + 1,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            model_name=gen_result.model_name if gen_result else "",
        )

    async def agenerate(
        self,
        pytorch_code: str,
        operation_desc: str = "",
        input_shapes: Optional[dict] = None,
    ) -> GenerationOutput:
        """Async variant of generate.

        Args:
            pytorch_code: Reference PyTorch implementation.
            operation_desc: Natural-language description of the operation.
            input_shapes: Optional mapping of input name to shape tuple.

        Returns:
            GenerationOutput with generated code, validity status, and metrics.
        """
        enriched_desc = self._build_enriched_operation_desc(operation_desc, input_shapes)

        total_tokens = 0
        total_latency = 0.0
        all_errors: list[str] = []
        gen_result: Optional[GenerationResult] = None

        for attempt in range(1, _MAX_PARSE_RETRIES + 2):
            logger.info("Async generation attempt %d/%d", attempt, _MAX_PARSE_RETRIES + 1)

            if attempt == 1:
                prompt = self.prompt_builder.build_generation_prompt(
                    pytorch_code=pytorch_code,
                    operation_desc=enriched_desc,
                    skill_memory=skill_memory,
                )
            else:
                prompt = self.prompt_builder.build_generation_prompt(
                    pytorch_code=pytorch_code,
                    operation_desc=enriched_desc + _FORMAT_RETRY_SUFFIX,
                    skill_memory=skill_memory,
                )

            gen_result = await self.llm_client.agenerate(
                prompt, system_message=self.system_prompt
            )
            total_tokens += gen_result.tokens_used
            total_latency += gen_result.latency_ms

            if gen_result.error:
                logger.error("LLM error on attempt %d: %s", attempt, gen_result.error)
                all_errors.append(f"LLM error: {gen_result.error}")
                continue

            parse_result: ParseResult = self.response_parser.parse(
                gen_result.code, original_pytorch_code=pytorch_code
            )

            if parse_result.is_valid:
                syntax_errors = self.response_parser.validate_syntax(parse_result.code)
                if syntax_errors:
                    logger.warning("Syntax errors on attempt %d: %s", attempt, syntax_errors)
                    all_errors.extend(syntax_errors)
                    continue

                logger.info("Valid kernel generated on attempt %d", attempt)
                return GenerationOutput(
                    code=parse_result.code,
                    is_valid=True,
                    parse_errors=[],
                    attempts=attempt,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency,
                    model_name=gen_result.model_name,
                )

            logger.warning("Parse failed on attempt %d: %s", attempt, parse_result.parse_errors)
            all_errors.extend(parse_result.parse_errors)

        logger.error("All async generation attempts failed. Errors: %s", all_errors)
        return GenerationOutput(
            code=None,
            is_valid=False,
            parse_errors=all_errors,
            attempts=_MAX_PARSE_RETRIES + 1,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            model_name=gen_result.model_name if gen_result else "",
        )
