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
    # Roofline view (AMK-style): speedup is relative to a baseline whose own
    # quality varies by hardware (H100 cuBLAS is much stronger than Blackwell's
    # current libraries), so we also report the absolute achieved HBM bandwidth
    # against the device peak. memory-bound lower bound = io_bytes / peak_bw.
    io_bytes: Optional[int] = None          # input + output tensor traffic (lower bound)
    achieved_gbs: Optional[float] = None    # io_bytes / triton_time
    hbm_util_pct: Optional[float] = None    # achieved / device peak * 100


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


# Spec peak HBM bandwidth (GB/s) by device-name substring. Used for the
# roofline view; override with LLM_KERNEL_PEAK_BW_GBS when the device is
# missing or a custom figure is preferred. Values are vendor spec sheets.
_PEAK_BW_GBS = {
    "RTX PRO 6000 Blackwell": 1792.0,
    "B200": 8000.0,
    "H200": 4800.0,
    "H100": 3350.0,   # SXM; PCIe is 2000
    "A100": 1555.0,   # 40GB; 80GB SXM is 2039
    "RTX A6000": 768.0,
    "L40S": 864.0,
    "RTX 5090": 1792.0,
    "RTX 4090": 1008.0,
}


def _device_peak_bw_gbs() -> Optional[float]:
    env = os.environ.get("LLM_KERNEL_PEAK_BW_GBS")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    try:
        import torch
        name = torch.cuda.get_device_name(0)
    except Exception:
        return None
    for key, bw in _PEAK_BW_GBS.items():
        if key in name:
            return bw
    return None


def _tensor_bytes(obj) -> int:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(x) for x in obj)
    return 0


def _cleanup_temp_dir(tmp_dir: str) -> None:
    """Clean up temporary directory after evaluation."""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass  # Ignore cleanup errors


def _chunked_compare(ref_t, gen_t, atol: float, rtol: float,
                     chunk_elems: int = 64_000_000) -> dict:
    """allclose + rich diagnostics computed over flat chunks.

    A full-tensor comparison materialises abs-diff, tolerance and mask tensors
    the same size as the output; on multi-GB outputs that triples peak VRAM and
    OOMs 48 GB cards, falsely failing correct kernels. Chunking caps the
    comparison overhead at ~3 chunk-sized temporaries (256 MB each).
    """
    import torch

    ref_v = ref_t.reshape(-1)
    gen_v = gen_t.reshape(-1)
    n_tot = ref_v.numel()
    max_abs = 0.0
    max_rel = 0.0
    n_bad = 0
    worst_ref = worst_gen = 0.0
    gen_absmax = 0.0
    ref_sum = gen_sum = 0.0
    ref_sumsq = gen_sumsq = 0.0
    for s in range(0, n_tot, chunk_elems):
        r = ref_v[s:s + chunk_elems].float()
        g = gen_v[s:s + chunk_elems].float()
        d = (r - g).abs()
        m = d.max().item()
        if m > max_abs:
            max_abs = m
            w = int(d.argmax().item())
            worst_ref = r[w].item()
            worst_gen = g[w].item()
        bad = d > (atol + rtol * r.abs())
        n_bad += int(bad.sum().item())
        del bad
        max_rel = max(max_rel, (d / r.abs().clamp_min(1e-12)).max().item())
        gen_absmax = max(gen_absmax, g.abs().max().item())
        ref_sum += r.sum().item(); gen_sum += g.sum().item()
        ref_sumsq += (r * r).sum().item(); gen_sumsq += (g * g).sum().item()
        del r, g, d
    ref_mean = ref_sum / max(n_tot, 1)
    gen_mean = gen_sum / max(n_tot, 1)
    ref_std = max(ref_sumsq / max(n_tot, 1) - ref_mean ** 2, 0.0) ** 0.5
    gen_std = max(gen_sumsq / max(n_tot, 1) - gen_mean ** 2, 0.0) ** 0.5
    return {
        "is_close": n_bad == 0,
        "max_abs": max_abs, "max_rel": max_rel,
        "n_bad": n_bad, "n_tot": n_tot,
        "worst_ref": worst_ref, "worst_gen": worst_gen,
        "gen_allzero": gen_absmax == 0.0,
        "ref_mean": ref_mean, "ref_std": ref_std,
        "gen_mean": gen_mean, "gen_std": gen_std,
    }


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

                # Inputs are no longer needed once both outputs exist; on
                # multi-GB problems (P19+ activations) keeping them alive
                # pushes the comparison below over the VRAM limit of 48 GB
                # cards even though the kernel itself is correct.
                del inputs
                torch.cuda.empty_cache()

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

                    cmp = _chunked_compare(ref_t, gen_t, self.atol, self.rtol)
                    trial_max = cmp["max_abs"]
                    max_abs_error = max(max_abs_error, trial_max)

                    if not cmp["is_close"]:
                        # Rich diagnostics so the refinement prompt can pinpoint
                        # the failure (scale error, all-zeros, dtype mismatch, ...).
                        dtype_note = (
                            f" | dtype differs (ref={ref_t.dtype}, gen={gen_t.dtype}) — "
                            f"output must match the reference dtype"
                            if ref_t.dtype != gen_t.dtype else ""
                        )
                        zero_note = (
                            " | generated output is ALL ZEROS (kernel likely not "
                            "writing results / wrong grid or mask)"
                            if cmp["gen_allzero"] else ""
                        )
                        err = (
                            f"Numerical mismatch on output {idx} (trial {trial}): "
                            f"max_abs_error={trial_max:.3e}, max_rel_error={cmp['max_rel']:.3e}, "
                            f"{cmp['n_bad']}/{cmp['n_tot']} "
                            f"({100.0 * cmp['n_bad'] / max(cmp['n_tot'], 1):.1f}%) elements "
                            f"exceed tolerance (atol={self.atol}, rtol={self.rtol}). "
                            f"Worst element: ref={cmp['worst_ref']:.6g} vs gen={cmp['worst_gen']:.6g}. "
                            f"ref[mean={cmp['ref_mean']:.4g}, std={cmp['ref_std']:.4g}], "
                            f"gen[mean={cmp['gen_mean']:.4g}, std={cmp['gen_std']:.4g}]"
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
                del ref_output, gen_output, ref_tensors, gen_tensors
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

            # Roofline view: io traffic lower bound = inputs + one output pass.
            io_bytes = achieved_gbs = hbm_util_pct = None
            try:
                with torch.no_grad():
                    out_probe = gen_model(*inputs)
                io_bytes = _tensor_bytes(inputs) + _tensor_bytes(out_probe)
                del out_probe
                if io_bytes and triton_ms and triton_ms > 0:
                    achieved_gbs = io_bytes / (triton_ms * 1e-3) / 1e9
                    peak = _device_peak_bw_gbs()
                    if peak:
                        hbm_util_pct = achieved_gbs / peak * 100.0
            except Exception:  # noqa: BLE001 — roofline is informational only
                pass

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
                io_bytes=io_bytes,
                achieved_gbs=achieved_gbs,
                hbm_util_pct=hbm_util_pct,
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
