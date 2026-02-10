"""Pipeline Orchestrator for Triton Kernel Generation.

Coordinates the full generate -> evaluate -> reflect/optimize loop,
managing state transitions, iteration history, and batch execution.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from llm_kernel.client.llm_client import LLMClient
from llm_kernel.client.prompt_builder import PromptBuilder
from llm_kernel.client.response_parser import ResponseParser
from llm_kernel.knowledge.gpu_specs import GPUSpec, get_gpu_spec, detect_current_gpu
from llm_kernel.pipeline.generator import KernelGenerator, GenerationOutput
from llm_kernel.pipeline.evaluator import KernelEvaluator, EvalResult
from llm_kernel.pipeline.reflector import Reflector, Feedback
from llm_kernel.pipeline.optimizer import PerformanceOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Record of a single pipeline iteration."""

    iteration: int
    code: Optional[str]
    eval_result: Optional[EvalResult]
    feedback: Optional[Feedback]
    tokens_used: int
    latency_ms: float


@dataclass
class PipelineResult:
    """Final result of the full pipeline for one problem."""

    problem_id: Optional[int]
    level: Optional[int]
    is_success: bool
    final_code: Optional[str]
    final_eval: Optional[EvalResult]
    iterations: int
    history: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    total_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Pipeline states
# ---------------------------------------------------------------------------

class _State:
    GENERATE = "GENERATE"
    EVALUATE = "EVALUATE"
    CLASSIFY = "CLASSIFY"
    REFLECT = "REFLECT"
    OPTIMIZE = "OPTIMIZE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class PipelineOrchestrator:
    """Orchestrates the iterative generate-evaluate-reflect-optimize pipeline.

    State machine per problem:
        GENERATE -> EVALUATE -> CLASSIFY ->
            (REFLECT -> GENERATE)   if compilation or correctness error
            (OPTIMIZE -> GENERATE)  if correct but slow
            COMPLETE                if correct and fast enough
            FAILED                  if max iterations exceeded
    """

    def __init__(self, config: dict):
        """Initialize all pipeline components from a config dictionary.

        The config schema matches ``config/default.yaml``.

        Args:
            config: Nested dictionary with keys ``gpu``, ``llm``,
                ``generation``, ``pipeline``, ``kernelbench``, ``execution``.
        """
        self.config = config

        # ----- GPU spec -----
        gpu_cfg = config.get("gpu", {})
        self.gpu_spec = self._resolve_gpu_spec(gpu_cfg)

        # ----- LLM client -----
        llm_cfg = config.get("llm", {})
        openai_cfg = config.get("openai", {})

        # Determine provider from config
        provider = llm_cfg.get("provider", "vllm")

        if provider == "openai":
            # OpenAI API configuration
            import os
            api_key = os.getenv(openai_cfg.get("api_key_env", "OPENAI_API_KEY"), "")
            model = openai_cfg.get("default_model", "gpt-4o")
            timeout = float(openai_cfg.get("timeout", 120))

            self.llm_client = LLMClient(
                provider="openai",
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
        else:
            # vLLM (local server) configuration
            self.llm_client = LLMClient(
                base_url=llm_cfg.get("base_url", "http://localhost:8000/v1"),
                api_key=llm_cfg.get("api_key", "EMPTY"),
                model=llm_cfg.get("default_model", "Qwen/Qwen2.5-Coder-32B-Instruct"),
                timeout=float(llm_cfg.get("timeout", 180)),
                provider="vllm",
            )

        # ----- Prompt builder & response parser -----
        prompts_dir = config.get("prompts_dir")
        self.prompt_builder = PromptBuilder(
            prompts_dir=Path(prompts_dir) if prompts_dir else None
        )
        self.response_parser = ResponseParser()

        # ----- Generator -----
        self.generator = KernelGenerator(
            llm_client=self.llm_client,
            prompt_builder=self.prompt_builder,
            response_parser=self.response_parser,
            gpu_spec=self.gpu_spec,
        )

        # ----- Evaluator -----
        pipe_cfg = config.get("pipeline", {})
        kb_cfg = config.get("kernelbench", {})
        self.evaluator = KernelEvaluator(
            gpu_arch=gpu_cfg.get("arch", "sm_120"),
            timeout=pipe_cfg.get("timeout_seconds", 300),
            n_correctness=kb_cfg.get("n_correctness", 5),
            n_timed=kb_cfg.get("n_trial", 10),
            atol=float(pipe_cfg.get("correctness_tolerance", 1e-2)),
        )

        # ----- Reflector & optimizer -----
        self.reflector = Reflector()
        self.optimizer = PerformanceOptimizer(gpu_spec=self.gpu_spec)

        # ----- Pipeline parameters -----
        self.max_iterations: int = pipe_cfg.get("max_iterations", 5)
        self.speedup_threshold: float = float(pipe_cfg.get("speedup_threshold", 1.0))
        self.strategy: str = pipe_cfg.get("strategy", "iterative")

        # ----- Execution -----
        exec_cfg = config.get("execution", {})
        self.max_workers: int = exec_cfg.get("max_workers", 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_single(
        self,
        pytorch_code: str,
        problem_id: Optional[int] = None,
        level: Optional[int] = None,
        operation_desc: str = "",
    ) -> PipelineResult:
        """Run the full pipeline for a single problem.

        Args:
            pytorch_code: PyTorch reference implementation (KernelBench format).
            problem_id: Optional problem identifier.
            level: Optional difficulty level.
            operation_desc: Natural-language description of the operation.

        Returns:
            PipelineResult with success status, final code, and history.
        """
        start_time = time.time()
        pid_tag = f"[problem={problem_id}] " if problem_id is not None else ""
        logger.info("%sStarting pipeline (strategy=%s, max_iter=%d)", pid_tag, self.strategy, self.max_iterations)

        history: list[IterationRecord] = []
        total_tokens = 0

        state = _State.GENERATE
        current_code: Optional[str] = None
        current_eval: Optional[EvalResult] = None
        current_feedback: Optional[Feedback] = None
        iteration = 0

        while state not in (_State.COMPLETE, _State.FAILED):
            # ----------------------------------------------------------
            # GENERATE
            # ----------------------------------------------------------
            if state == _State.GENERATE:
                iteration += 1
                if iteration > self.max_iterations:
                    logger.warning("%sMax iterations (%d) exceeded", pid_tag, self.max_iterations)
                    state = _State.FAILED
                    continue

                logger.info("%sIteration %d: GENERATE", pid_tag, iteration)

                if iteration == 1:
                    # Initial generation
                    gen_output: GenerationOutput = self.generator.generate(
                        pytorch_code=pytorch_code,
                        operation_desc=operation_desc,
                    )
                else:
                    # Refinement generation using feedback
                    assert current_feedback is not None
                    feedback_text = self._format_feedback(current_feedback)

                    if current_feedback.error_type == "performance" and current_code:
                        # Optimization prompt
                        profile_data = None
                        if current_eval and current_eval.speedup is not None:
                            profile_data = {
                                "speedup": current_eval.speedup,
                                "pytorch_time_ms": current_eval.pytorch_time_ms,
                                "triton_time_ms": current_eval.triton_time_ms,
                            }
                        prompt = self.prompt_builder.build_optimization_prompt(
                            pytorch_code=pytorch_code,
                            current_code=current_code,
                            profile_data=profile_data,
                        )
                    else:
                        # Refinement prompt
                        prompt = self.prompt_builder.build_refinement_prompt(
                            pytorch_code=pytorch_code,
                            prev_code=current_code or "",
                            error_feedback=feedback_text,
                            iteration=iteration,
                        )

                    llm_result = self.llm_client.generate(prompt)
                    total_tokens += llm_result.tokens_used

                    if llm_result.error:
                        logger.error("%sLLM error: %s", pid_tag, llm_result.error)
                        # Record and retry
                        history.append(IterationRecord(
                            iteration=iteration,
                            code=None,
                            eval_result=None,
                            feedback=Feedback(
                                error_type="compilation",
                                summary=f"LLM error: {llm_result.error}",
                                details=llm_result.error,
                                suggestions=["Retry generation."],
                            ),
                            tokens_used=llm_result.tokens_used,
                            latency_ms=llm_result.latency_ms,
                        ))
                        state = _State.GENERATE
                        continue

                    parse_result = self.response_parser.parse(
                        llm_result.code, original_pytorch_code=pytorch_code
                    )

                    gen_output = GenerationOutput(
                        code=parse_result.code if parse_result.is_valid else None,
                        is_valid=parse_result.is_valid,
                        parse_errors=parse_result.parse_errors,
                        attempts=1,
                        total_tokens=llm_result.tokens_used,
                        total_latency_ms=llm_result.latency_ms,
                        model_name=llm_result.model_name,
                    )

                total_tokens += gen_output.total_tokens

                if not gen_output.is_valid or gen_output.code is None:
                    logger.warning("%sGeneration failed: %s", pid_tag, gen_output.parse_errors)
                    current_feedback = Feedback(
                        error_type="compilation",
                        summary="Code generation/parsing failed.",
                        details="\n".join(gen_output.parse_errors),
                        suggestions=["Produce valid Python with Model class, @triton.jit kernel, get_inputs(), get_init_inputs()."],
                    )
                    history.append(IterationRecord(
                        iteration=iteration,
                        code=None,
                        eval_result=None,
                        feedback=current_feedback,
                        tokens_used=gen_output.total_tokens,
                        latency_ms=gen_output.total_latency_ms,
                    ))
                    state = _State.GENERATE
                    continue

                current_code = gen_output.code
                state = _State.EVALUATE

            # ----------------------------------------------------------
            # EVALUATE
            # ----------------------------------------------------------
            elif state == _State.EVALUATE:
                logger.info("%sIteration %d: EVALUATE", pid_tag, iteration)

                assert current_code is not None
                current_eval = self.evaluator.evaluate(
                    generated_code=current_code,
                    reference_code=pytorch_code,
                    problem_id=problem_id,
                )
                state = _State.CLASSIFY

            # ----------------------------------------------------------
            # CLASSIFY
            # ----------------------------------------------------------
            elif state == _State.CLASSIFY:
                logger.info("%sIteration %d: CLASSIFY (correct=%s, speedup=%s)",
                            pid_tag, iteration,
                            current_eval.is_correct if current_eval else "N/A",
                            f"{current_eval.speedup:.2f}" if current_eval and current_eval.speedup else "N/A")

                assert current_eval is not None

                if current_eval.compile_error or current_eval.runtime_error and not current_eval.is_correct:
                    # Compilation or runtime error -> reflect
                    current_feedback = self.reflector.analyze(
                        current_code or "", current_eval, iteration
                    )
                    history.append(IterationRecord(
                        iteration=iteration,
                        code=current_code,
                        eval_result=current_eval,
                        feedback=current_feedback,
                        tokens_used=0,
                        latency_ms=0.0,
                    ))
                    state = _State.REFLECT

                elif not current_eval.is_correct:
                    # Correctness error -> reflect
                    current_feedback = self.reflector.analyze(
                        current_code or "", current_eval, iteration
                    )
                    history.append(IterationRecord(
                        iteration=iteration,
                        code=current_code,
                        eval_result=current_eval,
                        feedback=current_feedback,
                        tokens_used=0,
                        latency_ms=0.0,
                    ))
                    state = _State.REFLECT

                elif current_eval.speedup is not None and current_eval.speedup >= self.speedup_threshold:
                    # Correct and fast enough -> success
                    history.append(IterationRecord(
                        iteration=iteration,
                        code=current_code,
                        eval_result=current_eval,
                        feedback=None,
                        tokens_used=0,
                        latency_ms=0.0,
                    ))
                    state = _State.COMPLETE

                else:
                    # Correct but slow -> optimize
                    current_feedback = self.optimizer.build_optimization_feedback(
                        current_code or "", current_eval
                    )
                    history.append(IterationRecord(
                        iteration=iteration,
                        code=current_code,
                        eval_result=current_eval,
                        feedback=current_feedback,
                        tokens_used=0,
                        latency_ms=0.0,
                    ))
                    state = _State.OPTIMIZE

            # ----------------------------------------------------------
            # REFLECT -> GENERATE
            # ----------------------------------------------------------
            elif state == _State.REFLECT:
                logger.info("%sIteration %d: REFLECT -> re-generating", pid_tag, iteration)
                state = _State.GENERATE

            # ----------------------------------------------------------
            # OPTIMIZE -> GENERATE
            # ----------------------------------------------------------
            elif state == _State.OPTIMIZE:
                logger.info("%sIteration %d: OPTIMIZE -> re-generating", pid_tag, iteration)
                state = _State.GENERATE

        # ----------------------------------------------------------
        # Build final result
        # ----------------------------------------------------------
        elapsed = time.time() - start_time
        is_success = state == _State.COMPLETE

        # If we failed but have a correct code from an earlier iteration, use it
        best_code = current_code
        best_eval = current_eval
        if not is_success:
            for rec in reversed(history):
                if rec.eval_result and rec.eval_result.is_correct and rec.code:
                    best_code = rec.code
                    best_eval = rec.eval_result
                    is_success = True
                    logger.info("%sFalling back to best correct code from iteration %d", pid_tag, rec.iteration)
                    break

        history_dicts = [
            {
                "iteration": rec.iteration,
                "code": rec.code,
                "eval_result": {
                    "is_correct": rec.eval_result.is_correct,
                    "speedup": rec.eval_result.speedup,
                    "compile_error": rec.eval_result.compile_error,
                    "runtime_error": rec.eval_result.runtime_error,
                    "correctness_error": rec.eval_result.correctness_error,
                    "pytorch_time_ms": rec.eval_result.pytorch_time_ms,
                    "triton_time_ms": rec.eval_result.triton_time_ms,
                    "max_abs_error": rec.eval_result.max_abs_error,
                } if rec.eval_result else None,
                "feedback": {
                    "error_type": rec.feedback.error_type,
                    "summary": rec.feedback.summary,
                    "details": rec.feedback.details,
                    "suggestions": rec.feedback.suggestions,
                } if rec.feedback else None,
            }
            for rec in history
        ]

        result = PipelineResult(
            problem_id=problem_id,
            level=level,
            is_success=is_success,
            final_code=best_code,
            final_eval=best_eval,
            iterations=iteration,
            history=history_dicts,
            total_tokens=total_tokens,
            total_time_seconds=elapsed,
        )

        logger.info(
            "%sPipeline complete: success=%s, iterations=%d, tokens=%d, time=%.1fs",
            pid_tag, is_success, iteration, total_tokens, elapsed,
        )
        return result

    def run_batch(
        self,
        problems: list[dict],
        max_workers: Optional[int] = None,
    ) -> list[PipelineResult]:
        """Run the pipeline on multiple problems using a thread pool.

        Each problem dict should have keys:
            - ``pytorch_code`` (str): Reference implementation.
            - ``problem_id`` (int, optional): Problem identifier.
            - ``level`` (int, optional): Difficulty level.
            - ``operation_desc`` (str, optional): Operation description.

        Args:
            problems: List of problem dictionaries.
            max_workers: Number of concurrent workers (defaults to config).

        Returns:
            List of PipelineResult, one per problem (order preserved).
        """
        workers = max_workers or self.max_workers
        logger.info("Starting batch run: %d problems, %d workers", len(problems), workers)

        results: list[Optional[PipelineResult]] = [None] * len(problems)
        futures_map: dict[Any, int] = {}

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, prob in enumerate(problems):
                future = executor.submit(
                    self.run_single,
                    pytorch_code=prob["pytorch_code"],
                    problem_id=prob.get("problem_id"),
                    level=prob.get("level"),
                    operation_desc=prob.get("operation_desc", ""),
                )
                futures_map[future] = idx

            for future in as_completed(futures_map):
                idx = futures_map[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("Problem %d failed with exception: %s", idx, exc)
                    results[idx] = PipelineResult(
                        problem_id=problems[idx].get("problem_id"),
                        level=problems[idx].get("level"),
                        is_success=False,
                        final_code=None,
                        final_eval=None,
                        iterations=0,
                        total_tokens=0,
                        total_time_seconds=0.0,
                    )

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_yaml(config_path: str) -> "PipelineOrchestrator":
        """Create an orchestrator from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configured PipelineOrchestrator instance.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return PipelineOrchestrator(config)

    def _resolve_gpu_spec(self, gpu_cfg: dict) -> GPUSpec:
        """Resolve GPU spec from config, registry, or auto-detect.

        Args:
            gpu_cfg: GPU section from the config.

        Returns:
            GPUSpec instance.
        """
        # Try registry lookup first
        gpu_name = gpu_cfg.get("registry_name")
        if gpu_name:
            spec = get_gpu_spec(gpu_name)
            if spec:
                logger.info("GPU spec loaded from registry: %s", spec.name)
                return spec

        # Try auto-detect
        detected = detect_current_gpu()
        if detected:
            logger.info("GPU auto-detected: %s", detected.name)
            return detected

        # Build from config values
        logger.info("Building GPU spec from config values")
        return GPUSpec(
            name=gpu_cfg.get("name", "Unknown GPU"),
            arch=gpu_cfg.get("arch", "sm_120"),
            compute_capability=gpu_cfg.get("compute_capability", "12.0"),
            vram_gb=gpu_cfg.get("vram_gb", 24),
            memory_bandwidth_gbs=float(gpu_cfg.get("memory_bandwidth_gbs", 1000)),
            fp16_tflops=float(gpu_cfg.get("fp16_tflops", 100)),
            bf16_tflops=float(gpu_cfg.get("bf16_tflops", gpu_cfg.get("fp16_tflops", 100))),
            fp32_tflops=float(gpu_cfg.get("fp32_tflops", gpu_cfg.get("fp16_tflops", 100) / 2)),
            cuda_cores=gpu_cfg.get("cuda_cores", 10000),
            tensor_cores=gpu_cfg.get("tensor_cores", 400),
            tensor_core_gen=gpu_cfg.get("tensor_core_gen", 5),
            tdp_watts=gpu_cfg.get("tdp_watts", 300),
        )

    @staticmethod
    def _format_feedback(feedback: Feedback) -> str:
        """Format Feedback into a string for the refinement prompt.

        Args:
            feedback: Feedback instance from reflector or optimizer.

        Returns:
            Formatted feedback string.
        """
        parts = [
            f"Error Type: {feedback.error_type}",
            f"Summary: {feedback.summary}",
            f"\nDetails:\n{feedback.details}",
        ]
        if feedback.suggestions:
            suggestions_text = "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(feedback.suggestions)
            )
            parts.append(f"\nSuggested Fixes:\n{suggestions_text}")
        return "\n".join(parts)
