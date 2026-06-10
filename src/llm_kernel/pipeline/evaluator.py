"""Kernel Evaluator for Generated Triton Code.

Evaluates generated Triton kernels against PyTorch reference implementations
by dynamically importing, checking correctness, and measuring performance.
"""

import importlib
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Number of random inputs for correctness checking
_DEFAULT_N_CORRECTNESS = 5
# Number of warmup runs before timing
_DEFAULT_N_WARMUP = 10
# Number of timed runs for performance measurement
_DEFAULT_N_TIMED = 100


@dataclass
class EvalResult:
    """Result of evaluating a generated Triton kernel."""

    is_correct: bool
    speedup: Optional[float] = None  # pytorch_time / triton_time (>1 means faster)
    compile_error: Optional[str] = None
    runtime_error: Optional[str] = None
    correctness_error: Optional[str] = None
    pytorch_time_ms: Optional[float] = None
    triton_time_ms: Optional[float] = None
    max_abs_error: Optional[float] = None


def _load_module_from_code(code: str, module_name: str) -> Tuple[Any, str]:
    """Write code to a temp file and dynamically import it.

    Args:
        code: Python source code.
        module_name: Name to assign to the loaded module.

    Returns:
        Tuple of (imported module object, temp directory path for cleanup).

    Raises:
        Exception: If the module cannot be compiled or imported.
    """
    tmp_dir = tempfile.mkdtemp(prefix="llm_kernel_eval_")
    file_path = os.path.join(tmp_dir, f"{module_name}.py")

    with open(file_path, "w") as f:
        f.write(code)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ImportError(f"Cannot create module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, tmp_dir


def _cleanup_temp_dir(tmp_dir: str) -> None:
    """Clean up temporary directory after evaluation."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass  # Ignore cleanup errors


def _cuda_timed_run(fn, inputs: tuple, n_warmup: int, n_timed: int) -> float:
    """Time a function using CUDA events for accurate GPU timing.

    Args:
        fn: Callable to time (e.g., model.forward).
        inputs: Tuple of input tensors.
        n_warmup: Number of warmup iterations.
        n_timed: Number of timed iterations.

    Returns:
        Average execution time in milliseconds.
    """
    import torch

    # Warmup
    for _ in range(n_warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_timed):
        fn(*inputs)
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    return elapsed_ms / n_timed


class KernelEvaluator:
    """Evaluates generated Triton kernels against PyTorch reference.

    Steps:
    1. Dynamically import the generated code.
    2. Run correctness checks with random inputs.
    3. Measure performance via CUDA event timing.
    """

    def __init__(
        self,
        gpu_arch: str = "sm_120",
        timeout: int = 300,
        n_correctness: int = _DEFAULT_N_CORRECTNESS,
        n_warmup: int = _DEFAULT_N_WARMUP,
        n_timed: int = _DEFAULT_N_TIMED,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ):
        """Initialize the evaluator.

        Args:
            gpu_arch: Target GPU architecture string.
            timeout: Maximum seconds for the entire evaluation.
            n_correctness: Number of random-input correctness trials.
            n_warmup: Warmup iterations before timing.
            n_timed: Timed iterations for performance measurement.
            atol: Absolute tolerance for correctness (torch.allclose).
            rtol: Relative tolerance for correctness.
        """
        self.gpu_arch = gpu_arch
        self.timeout = timeout
        self.n_correctness = n_correctness
        self.n_warmup = n_warmup
        self.n_timed = n_timed
        self.atol = atol
        self.rtol = rtol

    def evaluate(
        self,
        generated_code: str,
        reference_code: str,
        problem_id: Optional[int] = None,
    ) -> EvalResult:
        """Evaluate a generated Triton kernel against a PyTorch reference.

        Args:
            generated_code: Generated Triton kernel source (KernelBench format).
            reference_code: PyTorch reference source (KernelBench format).
            problem_id: Optional problem identifier for logging.

        Returns:
            EvalResult with correctness, performance, and error details.
        """
        import torch

        pid_tag = f"[problem={problem_id}] " if problem_id is not None else ""
        logger.info("%sStarting evaluation", pid_tag)

        # Track temp directories for cleanup
        temp_dirs = []

        # ------------------------------------------------------------------
        # Step 1: Compile/import the reference module
        # ------------------------------------------------------------------
        try:
            ref_module, ref_tmp_dir = _load_module_from_code(reference_code, f"_ref_{problem_id or 0}")
            temp_dirs.append(ref_tmp_dir)
        except Exception as exc:
            msg = f"Reference code import failed: {traceback.format_exc()}"
            logger.error("%s%s", pid_tag, msg)
            return EvalResult(
                is_correct=False,
                compile_error=f"Reference import error: {exc}",
            )

        # ------------------------------------------------------------------
        # Step 2: Compile/import the generated module
        # ------------------------------------------------------------------
        try:
            gen_module, gen_tmp_dir = _load_module_from_code(generated_code, f"_gen_{problem_id or 0}")
            temp_dirs.append(gen_tmp_dir)
        except SyntaxError as exc:
            msg = f"Syntax error in generated code: {exc}"
            logger.error("%s%s", pid_tag, msg)
            for d in temp_dirs:
                _cleanup_temp_dir(d)
            return EvalResult(is_correct=False, compile_error=msg)
        except Exception as exc:
            msg = f"Import error in generated code: {traceback.format_exc()}"
            logger.error("%s%s", pid_tag, msg)
            for d in temp_dirs:
                _cleanup_temp_dir(d)
            return EvalResult(is_correct=False, compile_error=str(exc))

        # ------------------------------------------------------------------
        # Step 3: Instantiate models
        # ------------------------------------------------------------------
        try:
            ref_init_inputs = ref_module.get_init_inputs()
            ref_model = ref_module.Model(*ref_init_inputs).cuda().eval()
        except Exception as exc:
            msg = f"Reference model init failed: {exc}"
            logger.error("%s%s", pid_tag, msg)
            for d in temp_dirs:
                _cleanup_temp_dir(d)
            return EvalResult(is_correct=False, runtime_error=msg)

        try:
            gen_init_inputs = gen_module.get_init_inputs()
            gen_model = gen_module.Model(*gen_init_inputs).cuda().eval()
        except Exception as exc:
            msg = f"Generated model init failed: {exc}"
            logger.error("%s%s", pid_tag, msg)
            for d in temp_dirs:
                _cleanup_temp_dir(d)
            return EvalResult(is_correct=False, runtime_error=msg)

        # ------------------------------------------------------------------
        # Step 4: Correctness check with multiple random inputs
        # ------------------------------------------------------------------
        max_abs_error = 0.0
        try:
            for trial in range(self.n_correctness):
                inputs = ref_module.get_inputs()
                inputs = tuple(
                    t.cuda() if isinstance(t, torch.Tensor) else t for t in inputs
                )

                with torch.no_grad():
                    ref_output = ref_model(*inputs)
                    gen_output = gen_model(*inputs)

                # Handle tuple/list outputs
                if isinstance(ref_output, (tuple, list)):
                    ref_tensors = ref_output
                    gen_tensors = gen_output if isinstance(gen_output, (tuple, list)) else (gen_output,)
                else:
                    ref_tensors = (ref_output,)
                    gen_tensors = (gen_output,)

                if len(ref_tensors) != len(gen_tensors):
                    err = (
                        f"Output count mismatch: reference has {len(ref_tensors)} "
                        f"outputs, generated has {len(gen_tensors)}"
                    )
                    logger.error("%s%s", pid_tag, err)
                    return EvalResult(
                        is_correct=False,
                        correctness_error=err,
                        max_abs_error=max_abs_error,
                    )

                for idx, (ref_t, gen_t) in enumerate(zip(ref_tensors, gen_tensors)):
                    if not isinstance(ref_t, torch.Tensor) or not isinstance(gen_t, torch.Tensor):
                        continue

                    if ref_t.shape != gen_t.shape:
                        err = (
                            f"Shape mismatch on output {idx}: "
                            f"reference={ref_t.shape}, generated={gen_t.shape}"
                        )
                        logger.error("%s%s", pid_tag, err)
                        return EvalResult(
                            is_correct=False,
                            correctness_error=err,
                            max_abs_error=max_abs_error,
                        )

                    ref_f, gen_f = ref_t.float(), gen_t.float()
                    abs_diff = (ref_f - gen_f).abs()
                    trial_max = abs_diff.max().item()
                    max_abs_error = max(max_abs_error, trial_max)

                    if not torch.allclose(
                        ref_f, gen_f, atol=self.atol, rtol=self.rtol
                    ):
                        # Rich diagnostics so the refinement prompt can pinpoint
                        # the failure (scale error, all-zeros, dtype mismatch, ...).
                        tol = self.atol + self.rtol * ref_f.abs()
                        bad = abs_diff > tol
                        n_bad, n_tot = int(bad.sum().item()), bad.numel()
                        max_rel = (abs_diff / ref_f.abs().clamp_min(1e-12)).max().item()
                        w = int(abs_diff.flatten().argmax().item())
                        rv = ref_f.flatten()[w].item()
                        gv = gen_f.flatten()[w].item()
                        dtype_note = (
                            f" | dtype differs (ref={ref_t.dtype}, gen={gen_t.dtype}) — "
                            f"output must match the reference dtype"
                            if ref_t.dtype != gen_t.dtype else ""
                        )
                        zero_note = (
                            " | generated output is ALL ZEROS (kernel likely not "
                            "writing results / wrong grid or mask)"
                            if gen_f.abs().max().item() == 0.0 else ""
                        )
                        err = (
                            f"Numerical mismatch on output {idx} (trial {trial}): "
                            f"max_abs_error={trial_max:.3e}, max_rel_error={max_rel:.3e}, "
                            f"{n_bad}/{n_tot} ({100.0 * n_bad / max(n_tot, 1):.1f}%) elements "
                            f"exceed tolerance (atol={self.atol}, rtol={self.rtol}). "
                            f"Worst element: ref={rv:.6g} vs gen={gv:.6g}. "
                            f"ref[mean={ref_f.mean().item():.4g}, std={ref_f.std().item():.4g}], "
                            f"gen[mean={gen_f.mean().item():.4g}, std={gen_f.std().item():.4g}]"
                            f"{dtype_note}{zero_note}"
                        )
                        logger.error("%s%s", pid_tag, err)
                        for d in temp_dirs:
                            _cleanup_temp_dir(d)
                        return EvalResult(
                            is_correct=False,
                            correctness_error=err,
                            max_abs_error=max_abs_error,
                        )

                # Free per-trial GPU tensors before the next iteration.
                # KernelBench inputs can be multi-GB (e.g. P19 ReLU is
                # 4096x393216 = 6 GB); without this, n_correctness trials
                # accumulate and OOM the GPU, falsely failing correct kernels.
                del ref_output, gen_output, inputs, ref_tensors, gen_tensors
                torch.cuda.empty_cache()

            logger.info(
                "%sCorrectness passed (%d trials, max_abs_error=%.6e)",
                pid_tag, self.n_correctness, max_abs_error,
            )

        except Exception as exc:
            msg = f"Runtime error during correctness check: {traceback.format_exc()}"
            logger.error("%s%s", pid_tag, msg)
            for d in temp_dirs:
                _cleanup_temp_dir(d)
            return EvalResult(
                is_correct=False,
                runtime_error=str(exc),
                max_abs_error=max_abs_error,
            )

        # ------------------------------------------------------------------
        # Step 5: Performance measurement
        # ------------------------------------------------------------------
        try:
            # Generate fresh inputs for timing
            inputs = ref_module.get_inputs()
            inputs = tuple(
                t.cuda() if isinstance(t, torch.Tensor) else t for t in inputs
            )

            pytorch_ms = _cuda_timed_run(
                ref_model, inputs, self.n_warmup, self.n_timed
            )
            triton_ms = _cuda_timed_run(
                gen_model, inputs, self.n_warmup, self.n_timed
            )

            speedup = pytorch_ms / triton_ms if triton_ms > 0 else 0.0

            logger.info(
                "%sPerformance: pytorch=%.3fms, triton=%.3fms, speedup=%.2fx",
                pid_tag, pytorch_ms, triton_ms, speedup,
            )

            # Cleanup temp directories
            for d in temp_dirs:
                _cleanup_temp_dir(d)

            return EvalResult(
                is_correct=True,
                speedup=speedup,
                pytorch_time_ms=pytorch_ms,
                triton_time_ms=triton_ms,
                max_abs_error=max_abs_error,
            )

        except Exception as exc:
            msg = f"Error during performance measurement: {traceback.format_exc()}"
            logger.warning("%s%s", pid_tag, msg)
            # Cleanup temp directories
            for d in temp_dirs:
                _cleanup_temp_dir(d)
            # Correctness passed but timing failed -- still report correct
            return EvalResult(
                is_correct=True,
                runtime_error=f"Timing error: {exc}",
                max_abs_error=max_abs_error,
            )
