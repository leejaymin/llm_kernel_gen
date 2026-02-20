"""
KernelBench Evaluation Runner.

Executes generated Triton kernels and compares against PyTorch reference
implementations for correctness and performance benchmarking.

Each kernel evaluation runs in a **separate subprocess** to prevent
CUDA error cascading (sticky errors that corrupt the CUDA context).
"""

import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any
import importlib.util
import traceback

import torch
import numpy as np


# --- Subprocess evaluation script (written to temp file and executed) ---
_SUBPROCESS_EVAL_SCRIPT = r'''
"""Subprocess kernel evaluator. Runs correctness + benchmark in isolation."""
import sys
import json
import time
import tempfile
import importlib.util
import traceback
from pathlib import Path

import torch
import numpy as np


def load_module(code: str, module_name: str):
    """Load Python code as a module from a temp file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        Path(temp_path).unlink(missing_ok=True)


def instantiate_model(module):
    """Instantiate Model class handling various init_inputs formats."""
    if hasattr(module, 'get_init_inputs'):
        init_inputs = module.get_init_inputs()
        if isinstance(init_inputs, dict):
            return module.Model(**init_inputs)
        elif isinstance(init_inputs, (list, tuple)) and len(init_inputs) > 0:
            return module.Model(*init_inputs)
        else:
            return module.Model()
    else:
        return module.Model()


def run_correctness(triton_module, reference_module, n_correctness, tolerance):
    """Test correctness of Triton kernel against reference."""
    if not hasattr(reference_module, 'get_inputs'):
        return False, "Reference code missing get_inputs() function"

    get_inputs = reference_module.get_inputs

    for trial in range(n_correctness):
        inputs = get_inputs()
        inputs_gpu = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                inputs_gpu.append(inp.cuda())
            else:
                inputs_gpu.append(inp)

        if hasattr(reference_module, 'Model'):
            ref_model = instantiate_model(reference_module).cuda()
            ref_output = ref_model(*inputs_gpu)
        else:
            return False, "Reference implementation structure not recognized"

        if hasattr(triton_module, 'Model'):
            triton_model = instantiate_model(triton_module).cuda()
            triton_output = triton_model(*inputs_gpu)
        else:
            return False, "Triton implementation missing Model class"

        if isinstance(ref_output, torch.Tensor):
            ref_output = [ref_output]
        if isinstance(triton_output, torch.Tensor):
            triton_output = [triton_output]

        for ref_out, triton_out in zip(ref_output, triton_output):
            if not isinstance(ref_out, torch.Tensor) or not isinstance(triton_out, torch.Tensor):
                continue
            if ref_out.shape != triton_out.shape:
                return False, f"Shape mismatch: {ref_out.shape} vs {triton_out.shape}"
            diff = torch.abs(ref_out - triton_out)
            max_diff = diff.max().item()
            rel_diff = (diff / (torch.abs(ref_out) + 1e-8)).max().item()
            if max_diff > tolerance and rel_diff > tolerance:
                return False, f"Value mismatch: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"

    return True, None


def run_benchmark(triton_module, reference_module, n_trial):
    """Benchmark Triton kernel vs reference."""
    get_inputs = reference_module.get_inputs

    inputs = get_inputs()
    inputs_gpu = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

    ref_model = instantiate_model(reference_module).cuda()
    triton_model = instantiate_model(triton_module).cuda()

    # Warmup
    for _ in range(3):
        ref_model(*inputs_gpu)
        triton_model(*inputs_gpu)
    torch.cuda.synchronize()

    # Benchmark reference
    ref_times = []
    for _ in range(n_trial):
        inputs = get_inputs()
        inputs_gpu = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        start = time.perf_counter()
        ref_model(*inputs_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        ref_times.append((end - start) * 1000)

    # Benchmark Triton
    triton_times = []
    for _ in range(n_trial):
        inputs = get_inputs()
        inputs_gpu = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]
        start = time.perf_counter()
        triton_model(*inputs_gpu)
        torch.cuda.synchronize()
        end = time.perf_counter()
        triton_times.append((end - start) * 1000)

    ref_time = float(np.median(ref_times))
    triton_time = float(np.median(triton_times))
    return triton_time, ref_time, None


def main():
    """Main entry point for subprocess evaluation."""
    # Read params from stdin
    params = json.loads(sys.stdin.read())

    generated_code = params['generated_code']
    reference_code = params['reference_code']
    problem_id = params['problem_id']
    level = params['level']
    n_correctness = params.get('n_correctness', 5)
    n_trial = params.get('n_trial', 10)
    tolerance = params.get('tolerance', 1e-2)

    result = {
        'problem_id': problem_id,
        'level': level,
        'correct': False,
        'speedup': 0.0,
        'compile_error': False,
        'runtime_error': False,
        'triton_time_ms': 0.0,
        'reference_time_ms': 0.0,
        'error_message': None,
    }

    # Load triton module
    try:
        triton_module = load_module(generated_code, f"triton_kernel_{problem_id}")
    except Exception as e:
        result['compile_error'] = True
        result['error_message'] = f"Triton compilation error: {str(e)}"
        print(json.dumps(result), flush=True)
        return

    # Load reference module
    try:
        reference_module = load_module(reference_code, f"reference_{problem_id}")
    except Exception as e:
        result['compile_error'] = True
        result['error_message'] = f"Reference compilation error: {str(e)}"
        print(json.dumps(result), flush=True)
        return

    # Correctness test
    try:
        is_correct, error_msg = run_correctness(triton_module, reference_module, n_correctness, tolerance)
    except Exception as e:
        is_correct = False
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    if not is_correct:
        result['runtime_error'] = True
        result['error_message'] = error_msg or "Correctness test failed"
        print(json.dumps(result), flush=True)
        return

    result['correct'] = True

    # Benchmark
    try:
        triton_time, ref_time, bench_error = run_benchmark(triton_module, reference_module, n_trial)
    except Exception as e:
        triton_time, ref_time = 0.0, 0.0
        bench_error = f"{type(e).__name__}: {str(e)}"

    if bench_error:
        result['runtime_error'] = True
        result['error_message'] = bench_error
        print(json.dumps(result), flush=True)
        return

    result['triton_time_ms'] = triton_time
    result['reference_time_ms'] = ref_time
    if triton_time > 0:
        result['speedup'] = ref_time / triton_time
    else:
        result['speedup'] = 0.0

    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
'''


class BenchRunner:
    """
    Run KernelBench evaluation for generated Triton kernels.

    Executes kernels in isolated subprocess environment to prevent CUDA
    error cascading between problems. Each evaluation gets a fresh CUDA
    context.
    """

    def __init__(
        self,
        backend: str = "triton",
        gpu_arch: str = "sm_120",
        n_correctness: int = 5,
        n_trial: int = 10,
        tolerance: float = 1e-2,
        subprocess_timeout: int = 120
    ):
        """
        Initialize benchmark runner.

        Args:
            backend: Backend to use ("triton" only supported currently)
            gpu_arch: Target GPU architecture (sm_90, sm_120, etc.)
            n_correctness: Number of correctness test iterations
            n_trial: Number of performance benchmark trials
            tolerance: Numerical tolerance for correctness validation
            subprocess_timeout: Timeout in seconds for each subprocess eval
        """
        self.backend = backend
        self.gpu_arch = gpu_arch
        self.n_correctness = n_correctness
        self.n_trial = n_trial
        self.tolerance = tolerance
        self.subprocess_timeout = subprocess_timeout

        # Validate GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. KernelBench requires GPU.")

        self.device = torch.device("cuda")

    @staticmethod
    def _instantiate_model(module: Any) -> Any:
        """
        Instantiate a Model class from a module, handling various init_inputs formats.

        KernelBench get_init_inputs() can return:
        - list/tuple of positional args (most common)
        - dict of keyword args
        - empty list/tuple/dict (no args)

        Args:
            module: Loaded Python module with Model class and optional get_init_inputs()

        Returns:
            Instantiated Model object
        """
        if hasattr(module, 'get_init_inputs'):
            init_inputs = module.get_init_inputs()
            if isinstance(init_inputs, dict):
                return module.Model(**init_inputs)
            elif isinstance(init_inputs, (list, tuple)) and len(init_inputs) > 0:
                return module.Model(*init_inputs)
            else:
                return module.Model()
        else:
            return module.Model()

    def _load_code_module(self, code: str, module_name: str) -> Any:
        """
        Dynamically load Python code as module.

        Args:
            code: Python source code
            module_name: Name for the module

        Returns:
            Loaded module

        Raises:
            Exception: If code fails to compile or load
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(module_name, temp_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            return module

        finally:
            # Cleanup temporary file
            Path(temp_path).unlink(missing_ok=True)

    def _run_in_subprocess(
        self,
        generated_code: str,
        reference_code: str,
        problem_id: int,
        level: int
    ) -> Dict[str, Any]:
        """
        Run kernel evaluation in a separate subprocess.

        This prevents CUDA error cascading - if one kernel causes an illegal
        memory access, it only affects the subprocess, not subsequent evaluations.

        Args:
            generated_code: Generated Triton kernel code
            reference_code: PyTorch reference implementation
            problem_id: Problem identifier
            level: Problem difficulty level

        Returns:
            Dictionary with evaluation results
        """
        default_result = {
            'problem_id': problem_id,
            'level': level,
            'correct': False,
            'speedup': 0.0,
            'compile_error': False,
            'runtime_error': False,
            'triton_time_ms': 0.0,
            'reference_time_ms': 0.0,
            'error_message': None,
        }

        # Write the eval script to a temp file
        script_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='_eval.py', delete=False
            ) as f:
                f.write(_SUBPROCESS_EVAL_SCRIPT)
                script_file = f.name

            # Prepare params as JSON to send via stdin
            params = {
                'generated_code': generated_code,
                'reference_code': reference_code,
                'problem_id': problem_id,
                'level': level,
                'n_correctness': self.n_correctness,
                'n_trial': self.n_trial,
                'tolerance': self.tolerance,
            }
            params_json = json.dumps(params)

            # Run subprocess with the same Python interpreter
            proc = subprocess.run(
                [sys.executable, script_file],
                input=params_json,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )

            # Parse JSON result from stdout
            stdout = proc.stdout.strip()
            if stdout:
                # Find the last line that looks like JSON (in case of stray prints)
                for line in reversed(stdout.split('\n')):
                    line = line.strip()
                    if line.startswith('{'):
                        try:
                            result = json.loads(line)
                            return result
                        except json.JSONDecodeError:
                            continue

            # If no valid JSON found, report error
            stderr_snippet = proc.stderr[-500:] if proc.stderr else ""
            default_result['runtime_error'] = True
            default_result['error_message'] = (
                f"Subprocess produced no valid JSON output.\n"
                f"stdout: {stdout[:500]}\n"
                f"stderr: {stderr_snippet}"
            )
            return default_result

        except subprocess.TimeoutExpired:
            default_result['runtime_error'] = True
            default_result['error_message'] = (
                f"Subprocess timed out after {self.subprocess_timeout}s"
            )
            return default_result

        except Exception as e:
            default_result['runtime_error'] = True
            default_result['error_message'] = (
                f"Subprocess execution error: {type(e).__name__}: {str(e)}"
            )
            return default_result

        finally:
            if script_file:
                Path(script_file).unlink(missing_ok=True)

    def run_single(
        self,
        generated_code: str,
        reference_code: str,
        problem_id: int,
        level: int
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single generated kernel in a subprocess.

        Each evaluation runs in a separate process to prevent CUDA error
        cascading between problems.

        Args:
            generated_code: Generated Triton kernel code
            reference_code: PyTorch reference implementation
            problem_id: Problem identifier
            level: Problem difficulty level

        Returns:
            Dictionary with evaluation results:
                - correct: bool - correctness status
                - speedup: float - speedup over reference (ref_time / triton_time)
                - compile_error: bool - whether compilation failed
                - runtime_error: bool - whether runtime error occurred
                - triton_time_ms: float - Triton execution time
                - reference_time_ms: float - Reference execution time
                - error_message: str - error details if any
                - problem_id: int
                - level: int
        """
        return self._run_in_subprocess(
            generated_code, reference_code, problem_id, level
        )

    def run_batch(
        self,
        problems: list[tuple[str, str, int, int]]
    ) -> list[Dict[str, Any]]:
        """
        Run evaluation on multiple problems.

        Each problem runs in its own subprocess for CUDA isolation.

        Args:
            problems: List of (generated_code, reference_code, problem_id, level) tuples

        Returns:
            List of result dictionaries
        """
        results = []

        for generated_code, reference_code, problem_id, level in problems:
            result = self.run_single(
                generated_code,
                reference_code,
                problem_id,
                level
            )
            results.append(result)

        return results

    def validate_environment(self) -> Dict[str, Any]:
        """
        Validate that environment is properly configured.

        Returns:
            Dictionary with environment validation results
        """
        info = {
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_name': None,
            'compute_capability': None,
            'triton_available': False,
        }

        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            info['compute_capability'] = f"sm_{capability[0]}{capability[1]}"

        # Check Triton availability
        try:
            import triton
            info['triton_available'] = True
        except ImportError:
            info['triton_available'] = False

        return info
