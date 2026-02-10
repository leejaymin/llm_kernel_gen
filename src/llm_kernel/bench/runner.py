"""
KernelBench Evaluation Runner.

Executes generated Triton kernels and compares against PyTorch reference
implementations for correctness and performance benchmarking.
"""

import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional, Any
import importlib.util
import traceback

import torch
import numpy as np


class BenchRunner:
    """
    Run KernelBench evaluation for generated Triton kernels.

    Executes kernels in isolated environment, validates correctness against
    PyTorch reference, and measures performance metrics.
    """

    def __init__(
        self,
        backend: str = "triton",
        gpu_arch: str = "sm_120",
        n_correctness: int = 5,
        n_trial: int = 10,
        tolerance: float = 1e-2
    ):
        """
        Initialize benchmark runner.

        Args:
            backend: Backend to use ("triton" only supported currently)
            gpu_arch: Target GPU architecture (sm_90, sm_120, etc.)
            n_correctness: Number of correctness test iterations
            n_trial: Number of performance benchmark trials
            tolerance: Numerical tolerance for correctness validation
        """
        self.backend = backend
        self.gpu_arch = gpu_arch
        self.n_correctness = n_correctness
        self.n_trial = n_trial
        self.tolerance = tolerance

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

    def _run_correctness_test(
        self,
        triton_module: Any,
        reference_module: Any,
    ) -> tuple[bool, Optional[str]]:
        """
        Test correctness of Triton kernel against PyTorch reference.

        Args:
            triton_module: Loaded Triton kernel module
            reference_module: Loaded PyTorch reference module

        Returns:
            Tuple of (is_correct, error_message)
        """
        try:
            # Get input generator
            if not hasattr(reference_module, 'get_inputs'):
                return False, "Reference code missing get_inputs() function"

            get_inputs = reference_module.get_inputs

            # Run multiple trials
            for trial in range(self.n_correctness):
                # Generate inputs
                inputs = get_inputs()

                # Ensure inputs are on GPU
                inputs_gpu = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        inputs_gpu.append(inp.cuda())
                    else:
                        inputs_gpu.append(inp)

                # Run reference implementation
                if hasattr(reference_module, 'Model'):
                    ref_model = self._instantiate_model(reference_module).cuda()
                    ref_output = ref_model(*inputs_gpu)
                else:
                    return False, "Reference implementation structure not recognized"

                # Run Triton implementation
                if hasattr(triton_module, 'Model'):
                    triton_model = self._instantiate_model(triton_module).cuda()
                    triton_output = triton_model(*inputs_gpu)
                else:
                    return False, "Triton implementation missing Model class"

                # Compare outputs
                if isinstance(ref_output, torch.Tensor):
                    ref_output = [ref_output]
                if isinstance(triton_output, torch.Tensor):
                    triton_output = [triton_output]

                # Validate each output tensor
                for ref_out, triton_out in zip(ref_output, triton_output):
                    if not isinstance(ref_out, torch.Tensor) or not isinstance(triton_out, torch.Tensor):
                        continue

                    # Check shapes match
                    if ref_out.shape != triton_out.shape:
                        return False, f"Shape mismatch: {ref_out.shape} vs {triton_out.shape}"

                    # Check values match within tolerance
                    diff = torch.abs(ref_out - triton_out)
                    max_diff = diff.max().item()
                    rel_diff = (diff / (torch.abs(ref_out) + 1e-8)).max().item()

                    if max_diff > self.tolerance and rel_diff > self.tolerance:
                        return False, f"Value mismatch: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e}"

            return True, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg

    def _benchmark_performance(
        self,
        triton_module: Any,
        reference_module: Any,
    ) -> tuple[float, float, Optional[str]]:
        """
        Benchmark Triton kernel vs PyTorch reference.

        Args:
            triton_module: Loaded Triton kernel module
            reference_module: Loaded PyTorch reference module

        Returns:
            Tuple of (triton_time_ms, reference_time_ms, error_message)
        """
        try:
            # Get input generator
            get_inputs = reference_module.get_inputs

            # Warmup
            inputs = get_inputs()
            inputs_gpu = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

            # Initialize models
            ref_model = self._instantiate_model(reference_module).cuda()
            triton_model = self._instantiate_model(triton_module).cuda()

            # Warmup runs
            for _ in range(3):
                ref_model(*inputs_gpu)
                triton_model(*inputs_gpu)

            torch.cuda.synchronize()

            # Benchmark reference
            ref_times = []
            for _ in range(self.n_trial):
                inputs = get_inputs()
                inputs_gpu = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

                start = time.perf_counter()
                ref_model(*inputs_gpu)
                torch.cuda.synchronize()
                end = time.perf_counter()

                ref_times.append((end - start) * 1000)  # Convert to ms

            # Benchmark Triton
            triton_times = []
            for _ in range(self.n_trial):
                inputs = get_inputs()
                inputs_gpu = [inp.cuda() if isinstance(inp, torch.Tensor) else inp for inp in inputs]

                start = time.perf_counter()
                triton_model(*inputs_gpu)
                torch.cuda.synchronize()
                end = time.perf_counter()

                triton_times.append((end - start) * 1000)

            # Compute median times
            ref_time = np.median(ref_times)
            triton_time = np.median(triton_times)

            return triton_time, ref_time, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return 0.0, 0.0, error_msg

    def run_single(
        self,
        generated_code: str,
        reference_code: str,
        problem_id: int,
        level: int
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single generated kernel.

        Args:
            generated_code: Generated Triton kernel code
            reference_code: PyTorch reference implementation
            problem_id: Problem identifier
            level: Problem difficulty level

        Returns:
            Dictionary with evaluation results:
                - correct: bool - correctness status
                - speedup: float - speedup over reference (triton_time / ref_time)
                - compile_error: bool - whether compilation failed
                - runtime_error: bool - whether runtime error occurred
                - triton_time_ms: float - Triton execution time
                - reference_time_ms: float - Reference execution time
                - error_message: str - error details if any
                - problem_id: int
                - level: int
        """
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

        # Load modules
        try:
            triton_module = self._load_code_module(
                generated_code,
                f"triton_kernel_{problem_id}"
            )
        except Exception as e:
            result['compile_error'] = True
            result['error_message'] = f"Triton compilation error: {str(e)}"
            return result

        try:
            reference_module = self._load_code_module(
                reference_code,
                f"reference_{problem_id}"
            )
        except Exception as e:
            result['compile_error'] = True
            result['error_message'] = f"Reference compilation error: {str(e)}"
            return result

        # Test correctness
        is_correct, error_msg = self._run_correctness_test(triton_module, reference_module)

        if not is_correct:
            result['runtime_error'] = True
            result['error_message'] = error_msg or "Correctness test failed"
            return result

        result['correct'] = True

        # Benchmark performance
        triton_time, ref_time, error_msg = self._benchmark_performance(
            triton_module,
            reference_module
        )

        if error_msg:
            result['runtime_error'] = True
            result['error_message'] = error_msg
            return result

        result['triton_time_ms'] = triton_time
        result['reference_time_ms'] = ref_time

        # Calculate speedup (higher is better)
        # Speedup = ref_time / triton_time
        if triton_time > 0:
            result['speedup'] = ref_time / triton_time
        else:
            result['speedup'] = 0.0

        return result

    def run_batch(
        self,
        problems: list[tuple[str, str, int, int]]
    ) -> list[Dict[str, Any]]:
        """
        Run evaluation on multiple problems.

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
