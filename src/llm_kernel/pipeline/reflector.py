"""Error Analysis and Feedback Generation for Triton Kernel Refinement.

Analyzes evaluation failures and produces structured feedback that guides
the LLM toward fixing compilation, correctness, or performance issues.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from llm_kernel.pipeline.evaluator import EvalResult
from llm_kernel.knowledge.triton_patterns import (
    get_error_solution,
    TRITON_COMMON_ERRORS,
)

logger = logging.getLogger(__name__)


@dataclass
class Feedback:
    """Structured feedback for the next generation attempt.

    Attributes:
        error_type: Category of the failure.
        summary: One-line human-readable summary.
        details: Multi-line detailed analysis.
        suggestions: Concrete fix suggestions for the LLM.
    """

    error_type: str  # "compilation" | "correctness" | "performance" | "timeout"
    summary: str
    details: str
    suggestions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Heuristic patterns for compilation error classification
# ---------------------------------------------------------------------------
_COMPILE_PATTERNS: list[tuple[str, str, list[str]]] = [
    (
        r"SyntaxError.*line (\d+)",
        "Python syntax error",
        [
            "Check for unclosed parentheses, brackets, or string literals.",
            "Ensure proper indentation throughout the file.",
            "Verify all function definitions are syntactically valid.",
        ],
    ),
    (
        r"NameError: name '(\w+)' is not defined",
        "Undefined variable or function",
        [
            "Ensure all variables are defined before use.",
            "Check that all required imports are present (torch, triton, triton.language as tl).",
            "Verify the @triton.jit kernel function is defined before the Model class calls it.",
        ],
    ),
    (
        r"ImportError|ModuleNotFoundError",
        "Missing import or module",
        [
            "Use only standard imports: torch, triton, triton.language as tl, torch.nn as nn.",
            "Do not import custom or third-party packages not available in the environment.",
        ],
    ),
    (
        r"triton.*(?:compile|MLIR|LLVM)",
        "Triton compilation error",
        [
            "Ensure block sizes are powers of 2 and declared as tl.constexpr.",
            "Check that tl.dot() operands have compatible shapes.",
            "Verify all pointer arithmetic is correct (offsets must be integer tensors).",
            "Use explicit dtype casting with .to(tl.float32) where needed.",
        ],
    ),
    (
        r"(?:out of bounds|illegal memory access|CUDA error)",
        "GPU memory access error",
        [
            "Add mask parameter to all tl.load() and tl.store() calls.",
            "Ensure offsets do not exceed tensor dimensions: mask = offsets < N.",
            "Check grid size calculation: use triton.cdiv(N, BLOCK_SIZE).",
        ],
    ),
    (
        r"constexpr|tl\.constexpr",
        "Missing constexpr annotation",
        [
            "All block size parameters must be typed as tl.constexpr.",
            "Example: def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Heuristic patterns for correctness error classification
# ---------------------------------------------------------------------------
_SHAPE_MISMATCH_RE = re.compile(r"[Ss]hape mismatch", re.IGNORECASE)
_NUMERICAL_RE = re.compile(r"[Nn]umerical|max_abs_error|allclose", re.IGNORECASE)


class Reflector:
    """Analyzes evaluation results and generates structured feedback."""

    def analyze(
        self,
        generated_code: str,
        eval_result: EvalResult,
        iteration: int,
    ) -> Feedback:
        """Analyze an evaluation failure and generate feedback for the next attempt.

        Args:
            generated_code: The Triton code that was evaluated.
            eval_result: Evaluation result from KernelEvaluator.
            iteration: Current iteration number (1-indexed).

        Returns:
            Feedback with error type, analysis, and fix suggestions.
        """
        if eval_result.compile_error:
            return self._analyze_compilation(
                generated_code, eval_result.compile_error, iteration
            )

        if eval_result.runtime_error and not eval_result.is_correct:
            return self._analyze_runtime(
                generated_code, eval_result.runtime_error, iteration
            )

        if eval_result.correctness_error:
            return self._analyze_correctness(
                generated_code, eval_result, iteration
            )

        if eval_result.is_correct and eval_result.speedup is not None and eval_result.speedup < 1.0:
            return self._analyze_performance(generated_code, eval_result, iteration)

        # Fallback: generic timeout / unknown
        return Feedback(
            error_type="timeout",
            summary=f"Evaluation did not complete within expected bounds (iteration {iteration}).",
            details=(
                "The kernel either timed out or produced an unclassified error. "
                "This may indicate an infinite loop, excessively large grid, or "
                "device synchronization deadlock."
            ),
            suggestions=[
                "Check grid dimensions -- ensure grid size is finite and proportional to input size.",
                "Avoid while-True loops; use bounded for-loops with known iteration count.",
                "Reduce block sizes if shared memory or register pressure is too high.",
                "Add tl.debug_barrier() calls to diagnose synchronization issues.",
            ],
        )

    # ------------------------------------------------------------------
    # Compilation errors
    # ------------------------------------------------------------------
    def _analyze_compilation(
        self, code: str, error_msg: str, iteration: int
    ) -> Feedback:
        """Analyze a compilation / import error."""
        suggestions: list[str] = []
        matched_category = "unknown compilation error"

        for pattern, category, fixes in _COMPILE_PATTERNS:
            if re.search(pattern, error_msg, re.IGNORECASE):
                matched_category = category
                suggestions.extend(fixes)
                break

        # Also search the Triton common-error database
        for err_type, solution in TRITON_COMMON_ERRORS.items():
            if err_type in error_msg.lower() or any(
                kw in error_msg.lower()
                for kw in err_type.replace("_", " ").split()
            ):
                suggestions.append(solution)

        # Extract line number if present
        line_match = re.search(r"line (\d+)", error_msg)
        line_hint = f" near line {line_match.group(1)}" if line_match else ""

        return Feedback(
            error_type="compilation",
            summary=f"Compilation failed: {matched_category}{line_hint} (iteration {iteration}).",
            details=(
                f"Error message:\n{error_msg}\n\n"
                "The generated code could not be compiled or imported. "
                "Review the error trace above and apply the suggested fixes."
            ),
            suggestions=suggestions or [
                "Carefully review the full traceback for the root cause.",
                "Ensure all imports are correct and complete.",
                "Check that the code follows valid Python 3.10+ syntax.",
            ],
        )

    # ------------------------------------------------------------------
    # Runtime errors
    # ------------------------------------------------------------------
    def _analyze_runtime(
        self, code: str, error_msg: str, iteration: int
    ) -> Feedback:
        """Analyze a runtime error (code imported but crashed during execution)."""
        suggestions: list[str] = [
            "Add boundary masks to all tl.load/tl.store: mask = offsets < N.",
            "Verify grid launch dimensions match the problem size.",
            "Check device placement -- all tensors must be on CUDA.",
        ]

        if "CUDA" in error_msg or "illegal" in error_msg.lower():
            suggestions.insert(
                0,
                "This is a GPU-side crash. Double-check pointer arithmetic and masking.",
            )

        if "shape" in error_msg.lower() or "size" in error_msg.lower():
            suggestions.append(
                "Verify output tensor allocation shape matches expected dimensions."
            )

        return Feedback(
            error_type="compilation",  # treat as fixable code bug
            summary=f"Runtime crash during execution (iteration {iteration}).",
            details=f"Error message:\n{error_msg}",
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Correctness errors
    # ------------------------------------------------------------------
    def _analyze_correctness(
        self, code: str, eval_result: EvalResult, iteration: int
    ) -> Feedback:
        """Analyze a correctness failure (shape or numerical mismatch)."""
        error_msg = eval_result.correctness_error or ""
        suggestions: list[str] = []

        if _SHAPE_MISMATCH_RE.search(error_msg):
            suggestions.extend([
                "Output shape must exactly match the PyTorch reference output shape.",
                "Check that the output tensor is allocated with the correct dimensions.",
                "Verify that the kernel writes to all required output positions.",
                "If the operation changes tensor rank (e.g., reduction), match that behaviour.",
            ])
            sub_type = "shape mismatch"
        else:
            abs_err = eval_result.max_abs_error
            abs_err_str = f"{abs_err:.6e}" if abs_err is not None else "unknown"
            suggestions.extend([
                f"Max absolute error was {abs_err_str} (tolerance is 1e-2).",
                "Use float32 accumulation even when inputs are float16/bfloat16.",
                "Check that reduction axes are correct (e.g., axis=0 vs axis=1).",
                "Ensure numerical stability: use log-sum-exp for softmax, stable mean/var for norms.",
                "Verify boundary masking -- unmasked loads may introduce garbage values.",
            ])
            sub_type = "numerical error"

        return Feedback(
            error_type="correctness",
            summary=f"Correctness failure ({sub_type}) on iteration {iteration}.",
            details=(
                f"Error details:\n{error_msg}\n\n"
                "The generated kernel produces outputs that differ from the "
                "PyTorch reference. Apply the fixes below."
            ),
            suggestions=suggestions,
        )

    # ------------------------------------------------------------------
    # Performance issues
    # ------------------------------------------------------------------
    def _analyze_performance(
        self, code: str, eval_result: EvalResult, iteration: int
    ) -> Feedback:
        """Analyze sub-par performance (correct but slower than PyTorch)."""
        speedup = eval_result.speedup or 0.0
        pt_ms = eval_result.pytorch_time_ms or 0.0
        tr_ms = eval_result.triton_time_ms or 0.0

        suggestions: list[str] = [
            "Add @triton.autotune with multiple (BLOCK_SIZE, num_warps) configs.",
            "Increase block sizes to improve memory coalescing and occupancy.",
            "Fuse multiple operations into a single kernel to reduce launch overhead.",
            "Check memory access patterns -- ensure coalesced reads and writes.",
            "Reduce register pressure by limiting per-thread working set.",
        ]

        # Heuristic: check for autotune
        if "@triton.autotune" not in code:
            suggestions.insert(0, "CRITICAL: Add @triton.autotune to automatically tune block sizes.")

        # Heuristic: check block sizes
        block_match = re.search(r"BLOCK_SIZE\s*[:=]\s*(\d+)", code)
        if block_match:
            bs = int(block_match.group(1))
            if bs < 128:
                suggestions.insert(0, f"Block size is {bs} -- try 128 or 256 for better GPU utilization.")

        return Feedback(
            error_type="performance",
            summary=(
                f"Kernel is correct but slower than PyTorch "
                f"(speedup={speedup:.2f}x, pytorch={pt_ms:.2f}ms, triton={tr_ms:.2f}ms) "
                f"on iteration {iteration}."
            ),
            details=(
                "The Triton kernel produces correct results but does not outperform "
                "the PyTorch baseline. Apply the optimization suggestions below."
            ),
            suggestions=suggestions,
        )
