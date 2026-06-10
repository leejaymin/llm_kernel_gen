"""
Jinja2-based Prompt Builder for Triton Kernel Generation.

Constructs prompts for different stages of kernel generation:
- Initial generation from PyTorch reference
- Refinement based on error feedback
- Optimization based on profiling data
"""

from pathlib import Path
from typing import Dict, Any, Optional

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound


class PromptBuilder:
    """
    Builds structured prompts for LLM kernel generation.

    Uses Jinja2 templates to construct context-aware prompts with:
    - GPU specifications
    - KernelBench output format requirements
    - Operation-specific guidance
    - Error feedback and iteration context
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt builder with template directory.

        Args:
            prompts_dir: Directory containing Jinja2 templates.
                        If None, uses project's prompts/ directory.
        """
        if prompts_dir is None:
            # Auto-detect prompts directory
            prompts_dir = Path(__file__).parent.parent.parent.parent / "prompts"

        self.prompts_dir = prompts_dir

        # Initialize Jinja2 environment
        if prompts_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(prompts_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            # Fallback to in-memory templates
            self.env = Environment()

        # GPU spec will be set per-project
        self.gpu_spec: Optional[Dict[str, Any]] = None

    def set_gpu_spec(self, gpu_spec: Dict[str, Any]) -> None:
        """
        Set GPU specification for prompts.

        Args:
            gpu_spec: Dictionary with keys:
                - name: GPU model name
                - arch: CUDA architecture (e.g., "sm_120")
                - vram_gb: VRAM size in GB
                - memory_bandwidth_gbs: Memory bandwidth in GB/s
                - fp16_tflops: FP16 TFLOPS
                - cuda_cores: Number of CUDA cores
                - tensor_cores: Number of Tensor cores
        """
        self.gpu_spec = gpu_spec

    def _get_template(self, template_name: str) -> Template:
        """
        Load Jinja2 template by name.

        Args:
            template_name: Template filename (e.g., "generation.j2")

        Returns:
            Jinja2 Template object
        """
        try:
            return self.env.get_template(template_name)
        except TemplateNotFound:
            # Return inline fallback template
            return self._get_fallback_template(template_name)

    def _get_fallback_template(self, template_name: str) -> Template:
        """
        Get inline fallback template if file not found.

        Args:
            template_name: Template identifier

        Returns:
            Jinja2 Template with inline content
        """
        fallbacks = {
            "generation.j2": self._generation_fallback(),
            "refinement.j2": self._refinement_fallback(),
            "optimization.j2": self._optimization_fallback(),
        }

        template_str = fallbacks.get(template_name, "")
        return self.env.from_string(template_str)

    def build_system_prompt(self) -> str:
        """Build system prompt for Triton kernel generation.

        Returns:
            System prompt string that constrains LLM output format.
        """
        return """You are an expert Triton GPU kernel programmer. Your ONLY job is to convert PyTorch code into optimized Triton GPU kernels.

STRICT OUTPUT RULES:
1. ALWAYS output ONLY valid Python code wrapped in a single ```python code block.
2. NEVER include explanations, comments or text outside the code block.
3. NO markdown headers, NO "Explanation:" sections, NO prose before or after the code block.

MANDATORY CODE STRUCTURE:
1. The code MUST contain `class Model(nn.Module)` — the class name MUST be exactly 'Model', NOT 'TritonModel' or any other name.
2. The Model class MUST inherit from `nn.Module` (i.e., `class Model(nn.Module):`, never `class Model:`).
3. The code MUST contain at least one `@triton.jit` decorated kernel function.
4. The code MUST contain a `def get_inputs()` function that returns sample input tensors ON CUDA device (use `.cuda()`).
5. The code MUST contain a `def get_init_inputs()` function (return empty tuple if no init args needed).
6. All required imports must be included: `import torch`, `import triton`, `import triton.language as tl`, `import torch.nn as nn`.

GPU RESOURCE CONSTRAINTS:
- Shared memory per block is ~100KB (101376 bytes). Choose BLOCK_SIZE conservatively:
  - For matmul/dot kernels: use BLOCK_SIZE=64 or smaller (128 will exceed shared memory).
  - For elementwise kernels: BLOCK_SIZE=1024 is fine.
- Use `num_warps=4` for small tiles, `num_warps=8` for larger tiles.
- Use `num_stages=2` by default. Reduce to 1 if shared memory is tight.

CRITICAL REMINDERS:
- get_inputs() MUST return tensors on CUDA device — always use `.cuda()` or `device='cuda'`.
- Do NOT include `if __name__ == '__main__'` blocks.
- Do NOT include test code, print statements, or benchmarking code.
- Return ONLY the ```python code block, nothing else."""

    def _generation_fallback(self) -> str:
        """Inline template for initial generation."""
        return """Convert the following PyTorch operation into an optimized Triton kernel.

**GPU Target**:
{% if gpu_spec %}
- Model: {{ gpu_spec.name }}
- Architecture: {{ gpu_spec.arch }}
- VRAM: {{ gpu_spec.vram_gb }} GB
- Memory Bandwidth: {{ gpu_spec.memory_bandwidth_gbs }} GB/s
- FP16 Performance: {{ gpu_spec.fp16_tflops }} TFLOPS
{% else %}
- General NVIDIA GPU
{% endif %}

**Operation Description**:
{{ operation_desc }}

**PyTorch Reference Code**:
```python
{{ pytorch_code }}
```

{% if skill_memory %}{{ skill_memory }}

{% endif %}**Required Output Format** (follow EXACTLY):
```python
import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def kernel_function(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = x  # your computation here
    tl.store(output_ptr + offsets, output, mask=mask)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        kernel_function[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output

def get_inputs():
    return [torch.randn(100, device='cuda')]

def get_init_inputs():
    return []
```

CRITICAL — INPUT CONTRACT (correctness is checked against this, read carefully):
- Your kernel is evaluated with the EXACT inputs produced by the reference `get_inputs()` above — typically **float32** on CUDA. Do NOT assume, require, cast, or `assert` a specific dtype such as float16/bfloat16. Never write `assert x.dtype in (...)` or `x.half()`.
- Accept whatever dtype the input tensors actually have. For matmul/reductions, accumulate in float32 (`acc = tl.zeros(..., dtype=tl.float32)`) and cast the result back to the input dtype only at the final store.
- `Model.forward` MUST accept the SAME number, shapes, and dtypes of arguments as the reference `forward`. Do not add, drop, reorder, or cast inputs.
- `get_inputs()` and `get_init_inputs()` MUST mirror the reference EXACTLY: same shapes and dtypes; `get_init_inputs()` returns the Model constructor argument list (often `[]`). Never return a dict like `{'forward': ...}`.

IMPORTANT RULES:
- class name MUST be exactly `Model` inheriting from `nn.Module`
- get_inputs() MUST return tensors on CUDA device
- Return ONLY Python code in a single ```python block
- Do NOT include any explanation text before or after the code
"""

    def _refinement_fallback(self) -> str:
        """Inline template for refinement."""
        return """
You are an expert in GPU kernel programming with Triton.

**Task**: Fix the following Triton kernel based on error feedback.

**Iteration**: {{ iteration }}

**GPU Target**:
{% if gpu_spec %}
- Model: {{ gpu_spec.name }}
- Architecture: {{ gpu_spec.arch }}
{% endif %}

**Original PyTorch Code**:
```python
{{ pytorch_code }}
```

**Previous Triton Implementation**:
```python
{{ prev_code }}
```

**Error Feedback**:
{{ error_feedback }}

**Instructions**:
1. Analyze the error carefully
2. Fix the issues while maintaining correctness
3. Ensure the output still follows KernelBench format (Model class, get_inputs, get_init_inputs)
4. Keep the same optimization strategy if possible

Return ONLY the corrected Python code, no explanations.
"""

    def _optimization_fallback(self) -> str:
        """Inline template for optimization."""
        return """
You are an expert in GPU kernel programming with Triton.

**Task**: Optimize the following Triton kernel based on profiling data.

**GPU Target**:
{% if gpu_spec %}
- Model: {{ gpu_spec.name }}
- Architecture: {{ gpu_spec.arch }}
- Memory Bandwidth: {{ gpu_spec.memory_bandwidth_gbs }} GB/s
- FP16 Performance: {{ gpu_spec.fp16_tflops }} TFLOPS
{% endif %}

**Original PyTorch Code**:
```python
{{ pytorch_code }}
```

**Current Triton Implementation**:
```python
{{ current_code }}
```

**Profiling Data**:
{% if profile_data %}
- Current Speedup: {{ profile_data.speedup }}x
- Memory Efficiency: {{ profile_data.memory_efficiency }}%
- Compute Utilization: {{ profile_data.compute_utilization }}%
- Bottleneck: {{ profile_data.bottleneck }}
{% endif %}

**Optimization Goals**:
1. Improve speedup over PyTorch baseline
2. Better utilize GPU memory bandwidth
3. Increase compute throughput
4. Reduce register pressure if needed

**Instructions**:
- Try different block sizes and num_warps configurations
- Consider memory access patterns (coalescing, bank conflicts)
- Explore loop unrolling and vectorization
- Maintain numerical correctness

Return ONLY the optimized Python code, no explanations.
"""

    def build_generation_prompt(
        self,
        pytorch_code: str,
        gpu_spec: Optional[Dict[str, Any]] = None,
        operation_desc: str = "",
        skill_memory: str = "",
    ) -> str:
        """
        Build prompt for initial kernel generation.

        Args:
            pytorch_code: Reference PyTorch implementation
            gpu_spec: GPU specifications (overrides instance gpu_spec)
            operation_desc: Natural language description of the operation

        Returns:
            Formatted prompt string
        """
        gpu_spec = gpu_spec or self.gpu_spec

        template = self._get_template("generation.j2")

        return template.render(
            pytorch_code=pytorch_code,
            gpu_spec=gpu_spec,
            operation_desc=operation_desc,
            skill_memory=skill_memory,
        )

    def build_refinement_prompt(
        self,
        pytorch_code: str,
        prev_code: str,
        error_feedback: str,
        iteration: int = 1,
        gpu_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build prompt for kernel refinement based on errors.

        Args:
            pytorch_code: Original PyTorch reference
            prev_code: Previous Triton implementation that failed
            error_feedback: Error message or compilation feedback
            iteration: Current iteration number
            gpu_spec: GPU specifications

        Returns:
            Formatted prompt string
        """
        gpu_spec = gpu_spec or self.gpu_spec

        template = self._get_template("refinement.j2")

        return template.render(
            pytorch_code=pytorch_code,
            prev_code=prev_code,
            error_feedback=error_feedback,
            iteration=iteration,
            gpu_spec=gpu_spec,
        )

    def build_optimization_prompt(
        self,
        pytorch_code: str,
        current_code: str,
        profile_data: Optional[Dict[str, Any]] = None,
        gpu_spec: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build prompt for kernel optimization based on profiling.

        Args:
            pytorch_code: Original PyTorch reference
            current_code: Current working Triton implementation
            profile_data: Profiling results with keys:
                - speedup: Current speedup vs PyTorch
                - memory_efficiency: Memory bandwidth utilization %
                - compute_utilization: Compute throughput %
                - bottleneck: Identified bottleneck (e.g., "memory", "compute")
            gpu_spec: GPU specifications

        Returns:
            Formatted prompt string
        """
        gpu_spec = gpu_spec or self.gpu_spec

        template = self._get_template("optimization.j2")

        return template.render(
            pytorch_code=pytorch_code,
            current_code=current_code,
            profile_data=profile_data,
            gpu_spec=gpu_spec,
        )

    def build_custom_prompt(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Build prompt from custom template.

        Args:
            template_name: Template filename in prompts directory
            **kwargs: Variables to pass to template

        Returns:
            Rendered prompt string
        """
        # Always include gpu_spec if available
        if self.gpu_spec and "gpu_spec" not in kwargs:
            kwargs["gpu_spec"] = self.gpu_spec

        template = self._get_template(template_name)
        return template.render(**kwargs)
