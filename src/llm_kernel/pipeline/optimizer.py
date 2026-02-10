"""Performance Optimizer for Generated Triton Kernels.

Performs static analysis of Triton code and generates optimization feedback
based on GPU specifications, code patterns, and optional profiling data.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from llm_kernel.knowledge.gpu_specs import GPUSpec
from llm_kernel.knowledge.blackwell_hints import (
    get_optimization_hints,
    get_general_hints,
)
from llm_kernel.pipeline.evaluator import EvalResult
from llm_kernel.pipeline.reflector import Feedback

logger = logging.getLogger(__name__)


@dataclass
class CodePatternAnalysis:
    """Result of static analysis on Triton kernel source code."""

    has_autotune: bool = False
    autotune_configs: int = 0
    block_sizes: dict[str, int] = field(default_factory=dict)
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    uses_tl_dot: bool = False
    uses_tl_load: bool = False
    uses_tl_store: bool = False
    uses_masking: bool = False
    has_shared_memory_hint: bool = False
    kernel_count: int = 0
    operation_types_detected: list[str] = field(default_factory=list)
    potential_issues: list[str] = field(default_factory=list)


class PerformanceOptimizer:
    """Analyzes Triton kernel code and generates optimization feedback.

    Combines static code analysis with GPU-specific knowledge to produce
    actionable suggestions for improving kernel performance.
    """

    def __init__(self, gpu_spec: GPUSpec):
        """Initialize with target GPU specification.

        Args:
            gpu_spec: Target GPU hardware specification.
        """
        self.gpu_spec = gpu_spec

    def analyze_code_patterns(self, code: str) -> dict[str, Any]:
        """Static analysis of Triton code for optimization opportunities.

        Scans the source for autotune decorators, block sizes, num_warps,
        memory access patterns, tensor core usage, and common anti-patterns.

        Args:
            code: Generated Triton kernel source code.

        Returns:
            Dictionary of analysis results (serializable).
        """
        analysis = self._analyze(code)
        return {
            "has_autotune": analysis.has_autotune,
            "autotune_configs": analysis.autotune_configs,
            "block_sizes": analysis.block_sizes,
            "num_warps": analysis.num_warps,
            "num_stages": analysis.num_stages,
            "uses_tl_dot": analysis.uses_tl_dot,
            "uses_tl_load": analysis.uses_tl_load,
            "uses_tl_store": analysis.uses_tl_store,
            "uses_masking": analysis.uses_masking,
            "has_shared_memory_hint": analysis.has_shared_memory_hint,
            "kernel_count": analysis.kernel_count,
            "operation_types_detected": analysis.operation_types_detected,
            "potential_issues": analysis.potential_issues,
        }

    def build_optimization_feedback(
        self,
        code: str,
        eval_result: EvalResult,
        profile_data: Optional[dict] = None,
    ) -> Feedback:
        """Analyze performance and suggest optimizations.

        Combines static code analysis with runtime evaluation results (and
        optional profiling data) to generate targeted optimization feedback.

        Args:
            code: Generated Triton kernel source code.
            eval_result: Evaluation result containing timing data.
            profile_data: Optional profiling data with keys like
                ``memory_efficiency``, ``compute_utilization``, ``bottleneck``.

        Returns:
            Feedback instance with optimization suggestions.
        """
        analysis = self._analyze(code)
        suggestions: list[str] = []
        details_parts: list[str] = []

        speedup = eval_result.speedup or 0.0
        pt_ms = eval_result.pytorch_time_ms or 0.0
        tr_ms = eval_result.triton_time_ms or 0.0

        details_parts.append(
            f"Current performance: pytorch={pt_ms:.3f}ms, triton={tr_ms:.3f}ms, "
            f"speedup={speedup:.2f}x"
        )

        # ----- Autotune -----
        if not analysis.has_autotune:
            suggestions.append(
                "Add @triton.autotune with at least 4 configurations varying "
                "BLOCK_SIZE, num_warps, and num_stages."
            )
        elif analysis.autotune_configs < 4:
            suggestions.append(
                f"Autotune has only {analysis.autotune_configs} configs. "
                "Add more (8-16 is typical) covering a wider range of block sizes "
                "and warp counts."
            )

        # ----- Block sizes -----
        for name, size in analysis.block_sizes.items():
            if size < 64:
                suggestions.append(
                    f"Block size {name}={size} is small. Try 128 or 256 for "
                    "better memory coalescing and higher occupancy."
                )
            elif size > 1024:
                suggestions.append(
                    f"Block size {name}={size} is very large. This may cause "
                    "register spilling. Consider reducing to 256-512."
                )

        if not analysis.block_sizes:
            suggestions.append(
                "No explicit block sizes detected. Define BLOCK_SIZE as a "
                "tl.constexpr parameter and tune it via @triton.autotune."
            )

        # ----- Num warps -----
        if analysis.num_warps is not None:
            if analysis.num_warps < 4:
                suggestions.append(
                    f"num_warps={analysis.num_warps} is low. Try 4 or 8 for "
                    "better latency hiding, especially on Blackwell."
                )
        else:
            suggestions.append(
                "num_warps not explicitly set. Add it to @triton.autotune configs "
                "to explore different warp configurations."
            )

        # ----- Masking -----
        if analysis.uses_tl_load and not analysis.uses_masking:
            suggestions.append(
                "Memory loads detected without masking. Add mask parameter to "
                "tl.load/tl.store for correctness and to avoid out-of-bounds access."
            )

        # ----- Tensor cores -----
        if not analysis.uses_tl_dot and self._likely_matmul(analysis):
            suggestions.append(
                "This appears to involve matrix multiplication but does not use "
                "tl.dot(). Use tl.dot() to leverage tensor cores for up to 10x speedup."
            )

        # ----- Profile-based analysis -----
        if profile_data:
            details_parts.append(f"Profiling data: {profile_data}")
            bottleneck = profile_data.get("bottleneck", "")
            mem_eff = profile_data.get("memory_efficiency", 0)
            compute_util = profile_data.get("compute_utilization", 0)

            if bottleneck == "memory" or (isinstance(mem_eff, (int, float)) and mem_eff < 50):
                suggestions.append(
                    f"Memory efficiency is low ({mem_eff}%). "
                    "Improve memory coalescing: ensure consecutive threads access "
                    "consecutive memory addresses. Consider tiling and shared memory staging."
                )
            if bottleneck == "compute" or (isinstance(compute_util, (int, float)) and compute_util < 50):
                suggestions.append(
                    f"Compute utilization is low ({compute_util}%). "
                    "Increase parallelism: use larger blocks, more warps, or fuse operations."
                )

        # ----- GPU-specific hints -----
        for op_type in analysis.operation_types_detected:
            op_hints = get_optimization_hints(op_type)
            if op_hints:
                # Pick the top 2 most relevant hints
                for hint in op_hints[:2]:
                    if hint not in suggestions:
                        suggestions.append(f"[{op_type}] {hint}")

        # ----- Add any detected issues -----
        for issue in analysis.potential_issues:
            if issue not in suggestions:
                suggestions.append(issue)

        # ----- General GPU hints -----
        general = get_general_hints()
        arch = self.gpu_spec.arch
        bw = self.gpu_spec.memory_bandwidth_gbs
        suggestions.append(
            f"Target GPU: {self.gpu_spec.name} ({arch}), "
            f"memory BW={bw} GB/s, FP16={self.gpu_spec.fp16_tflops} TFLOPS. "
            "Ensure kernel exploits these capabilities."
        )

        summary = (
            f"Optimization feedback: speedup={speedup:.2f}x "
            f"(target >1.0x). {len(suggestions)} suggestions generated."
        )

        return Feedback(
            error_type="performance",
            summary=summary,
            details="\n".join(details_parts),
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze(self, code: str) -> CodePatternAnalysis:
        """Run full static analysis on kernel source code."""
        a = CodePatternAnalysis()

        # Autotune
        autotune_matches = re.findall(r"@triton\.autotune", code)
        a.has_autotune = len(autotune_matches) > 0
        config_matches = re.findall(r"triton\.Config\s*\(", code)
        a.autotune_configs = len(config_matches)

        # Block sizes
        for m in re.finditer(r"(BLOCK_\w+)\s*[:=]\s*(\d+)", code):
            a.block_sizes[m.group(1)] = int(m.group(2))
        # Also check constexpr defaults
        for m in re.finditer(r"(BLOCK_\w+)\s*:\s*tl\.constexpr\s*=\s*(\d+)", code):
            a.block_sizes[m.group(1)] = int(m.group(2))

        # num_warps / num_stages
        nw_match = re.search(r"num_warps\s*[=:]\s*(\d+)", code)
        if nw_match:
            a.num_warps = int(nw_match.group(1))
        ns_match = re.search(r"num_stages\s*[=:]\s*(\d+)", code)
        if ns_match:
            a.num_stages = int(ns_match.group(1))

        # Triton API usage
        a.uses_tl_dot = "tl.dot(" in code
        a.uses_tl_load = "tl.load(" in code
        a.uses_tl_store = "tl.store(" in code
        a.uses_masking = bool(re.search(r"tl\.(load|store)\s*\([^)]*mask\s*=", code))

        # Shared memory hint (tl.zeros with shared)
        a.has_shared_memory_hint = "shared" in code.lower() and "tl." in code

        # Kernel count
        a.kernel_count = len(re.findall(r"@triton\.jit", code))

        # Operation type detection
        ops_detected: list[str] = []
        if "tl.dot(" in code:
            ops_detected.append("matmul")
        if re.search(r"tl\.(sum|max|min)\s*\(", code):
            ops_detected.append("reduction")
        if "softmax" in code.lower() or ("tl.exp(" in code and "tl.sum(" in code):
            ops_detected.append("softmax")
        if "attention" in code.lower() or ("query" in code.lower() and "key" in code.lower()):
            ops_detected.append("attention")
        if re.search(r"layer.?norm|batch.?norm|group.?norm", code, re.IGNORECASE):
            ops_detected.append("normalization")
        if re.search(r"conv\d|convolution", code, re.IGNORECASE):
            ops_detected.append("convolution")
        if not ops_detected:
            ops_detected.append("elementwise")
        a.operation_types_detected = ops_detected

        # Potential issues
        issues: list[str] = []
        if a.uses_tl_load and not a.uses_masking:
            issues.append(
                "Memory loads without mask parameter detected. "
                "This risks out-of-bounds access for non-power-of-2 sizes."
            )
        if a.kernel_count > 3:
            issues.append(
                f"Found {a.kernel_count} @triton.jit kernels. Consider fusing "
                "kernels to reduce launch overhead."
            )
        if "torch.autograd" in code:
            issues.append(
                "torch.autograd detected in generated code. For inference-only "
                "evaluation, remove autograd to reduce overhead."
            )
        # Check for Python-level loops over elements (anti-pattern)
        if re.search(r"for\s+\w+\s+in\s+range\s*\(\s*N\s*\)", code):
            issues.append(
                "Python-level loop over N elements detected. Move this to the "
                "Triton kernel for GPU parallelism."
            )
        a.potential_issues = issues

        return a

    def _likely_matmul(self, analysis: CodePatternAnalysis) -> bool:
        """Heuristic: does the code look like it involves matrix multiplication?"""
        return "matmul" in analysis.operation_types_detected or any(
            "BLOCK_M" in k or "BLOCK_N" in k or "BLOCK_K" in k
            for k in analysis.block_sizes
        )
