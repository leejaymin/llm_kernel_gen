"""
LLM Kernel CLI.

Command-line interface for Triton kernel generation and benchmarking
using local LLMs with vLLM inference server.
"""

import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import Optional, List
import json

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Project imports will be relative to package
# Assuming this runs as: python -m llm_kernel.cli
# or after installation: llm-kernel command

app = typer.Typer(
    name="llm-kernel",
    help="Local LLM Triton Kernel Generator for KernelBench",
    add_completion=False
)

console = Console()


def get_project_root() -> Path:
    """Get project root directory."""
    # Assuming cli.py is in src/llm_kernel/
    return Path(__file__).parent.parent.parent


def load_models_config() -> dict:
    """Load models configuration from YAML."""
    config_path = get_project_root() / "config" / "model" / "models.yaml"

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return {}

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return data.get("models", {})


def load_default_config() -> dict:
    """Load default configuration."""
    config_path = get_project_root() / "config" / "default.yaml"

    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_vllm_process() -> Optional[int]:
    """Find running vLLM server process."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().split()[0])

    except Exception:
        pass

    return None


@app.command()
def run(
    level: int = typer.Option(1, "--level", "-l", help="KernelBench level (1-4)"),
    problem_id: Optional[int] = typer.Option(None, "--problem", "-p", help="Specific problem ID (runs all if not specified)"),
    model: str = typer.Option("qwen2.5-coder-32b", "--model", "-m", help="Model name or alias"),
    strategy: str = typer.Option("iterative", "--strategy", "-s", help="Generation strategy (single_pass|iterative|best_of_n)"),
    max_iterations: int = typer.Option(5, "--iterations", "-i", help="Max refinement iterations"),
    num_samples: int = typer.Option(1, "--samples", "-n", help="Samples per problem (best-of-N)"),
    run_name: Optional[str] = typer.Option(None, "--name", help="Run name for results"),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Parallel workers"),
    base_url: str = typer.Option("http://localhost:8000/v1", "--url", help="LLM server URL"),
    provider: str = typer.Option("vllm", "--provider", help="LLM provider: vllm (local) or openai (commercial API)"),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", help="OpenAI API key (alternative to OPENAI_API_KEY env var)"),
    output_dir: str = typer.Option("runs", "--output", "-o", help="Output directory for results"),
):
    """Generate and evaluate Triton kernels on KernelBench."""

    console.print(Panel.fit(
        f"[bold cyan]LLM Kernel Benchmark Runner[/bold cyan]\n\n"
        f"Level: {level}\n"
        f"Problem: {problem_id if problem_id else 'All'}\n"
        f"Model: {model}\n"
        f"Provider: {provider}\n"
        f"Strategy: {strategy}",
        title="Configuration"
    ))

    try:
        # Import here to avoid circular imports
        from llm_kernel.bench.problem_loader import ProblemLoader
        from llm_kernel.bench.submission import SubmissionFormatter
        from llm_kernel.bench.runner import BenchRunner
        from llm_kernel.bench.results import ResultsAggregator
        from llm_kernel.client.llm_client import LLMClient
        from llm_kernel.client.prompt_builder import PromptBuilder
        from llm_kernel.client.response_parser import ResponseParser

        # Initialize components
        console.print("[cyan]Initializing components...[/cyan]")

        loader = ProblemLoader(dataset_src="huggingface")
        formatter = SubmissionFormatter()
        runner = BenchRunner()
        aggregator = ResultsAggregator(console=console)
        prompt_builder = PromptBuilder()
        response_parser = ResponseParser()

        # Load GPU spec from config and set on prompt builder
        config = load_default_config()
        gpu_spec = config.get('gpu', {})
        if gpu_spec:
            prompt_builder.set_gpu_spec(gpu_spec)

        # Build system prompt once (reused for all problems)
        system_prompt = prompt_builder.build_system_prompt()

        # Initialize LLM client with provider-specific settings
        if provider == "openai":
            llm_client = LLMClient(
                provider="openai",
                model=model,
                api_key=openai_api_key or "",  # Will use env var if not provided
                timeout=120.0,
            )
        else:
            llm_client = LLMClient(
                base_url=base_url,
                model=model,
                provider="vllm",
            )

        # Validate environment
        env_info = runner.validate_environment()
        if not env_info['cuda_available']:
            console.print("[red]Error: CUDA is not available[/red]")
            raise typer.Exit(1)

        if not env_info['triton_available']:
            console.print("[yellow]Warning: Triton not detected[/yellow]")

        console.print(f"[green]✓ GPU: {env_info['gpu_name']} ({env_info['compute_capability']})[/green]")

        # Check LLM server health (skip for OpenAI — always available)
        if provider != "openai":
            if not llm_client.health_check():
                console.print(f"[red]Error: LLM server not responding at {base_url}[/red]")
                console.print("[yellow]Start server with: llm-kernel server start[/yellow]")
                raise typer.Exit(1)

        console.print(f"[green]✓ LLM {'API' if provider == 'openai' else 'server'} ready[/green]")

        # Load problems
        console.print(f"[cyan]Loading problems from level {level}...[/cyan]")

        if problem_id is not None:
            problems = [loader.load_problem(level, problem_id)]
        else:
            problems = loader.load_level(level)

        console.print(f"[green]✓ Loaded {len(problems)} problems[/green]")

        # Generate run name
        if run_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_name = f"{model}_L{level}_{strategy}_{timestamp}"

        # Create output directory
        output_path = Path(output_dir) / run_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Run benchmark
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            task = progress.add_task(
                f"[cyan]Generating and evaluating kernels...",
                total=len(problems)
            )

            for idx, problem in enumerate(problems, 1):
                progress.update(
                    task,
                    description=f"[cyan]Problem {idx}/{len(problems)}: {problem.name}",
                    completed=idx-1
                )

                # Iterative refinement loop
                best_result = None
                prev_code = None
                error_feedback = None

                for iteration in range(1, max_iterations + 1):
                    # Build prompt
                    if iteration == 1 or error_feedback is None:
                        # Initial generation
                        prompt = prompt_builder.build_generation_prompt(
                            pytorch_code=problem.pytorch_code,
                            operation_desc=problem.operation_description,
                        )
                    else:
                        # Refinement with error feedback
                        prompt = prompt_builder.build_refinement_prompt(
                            pytorch_code=problem.pytorch_code,
                            prev_code=prev_code,
                            error_feedback=error_feedback,
                            iteration=iteration,
                        )

                    # Generate code with system message
                    generation_result = llm_client.generate(
                        prompt=prompt,
                        system_message=system_prompt,
                        temperature=0.3 + (iteration - 1) * 0.05,  # Slightly increase temp on retries
                        max_tokens=8192,
                    )

                    if generation_result.error:
                        if iteration == max_iterations:
                            console.print(f"[red]Generation failed for P{problem.problem_id}: {generation_result.error}[/red]")
                            best_result = {
                                'problem_id': problem.problem_id,
                                'level': level,
                                'correct': False,
                                'compile_error': True,
                                'error_message': generation_result.error,
                                'speedup': 0.0,
                            }
                        continue

                    # Parse response
                    parse_result = response_parser.parse(
                        generation_result.code,
                        original_pytorch_code=problem.pytorch_code,
                    )

                    if parse_result.warnings:
                        for w in parse_result.warnings:
                            console.print(f"  [yellow]⚠ P{problem.problem_id} iter{iteration}: {w}[/yellow]")

                    if not parse_result.is_valid:
                        error_feedback = "; ".join(parse_result.parse_errors)
                        prev_code = generation_result.code
                        if iteration < max_iterations:
                            console.print(f"  [yellow]↻ P{problem.problem_id} iter{iteration}: parse error, retrying...[/yellow]")
                            continue
                        best_result = {
                            'problem_id': problem.problem_id,
                            'level': level,
                            'correct': False,
                            'compile_error': True,
                            'error_message': f"Parse error: {error_feedback}",
                            'speedup': 0.0,
                        }
                        break

                    # Format for KernelBench submission
                    try:
                        formatted_code = formatter.format(parse_result.code, problem.pytorch_code)
                    except Exception as e:
                        error_feedback = f"Formatting error: {str(e)}"
                        prev_code = parse_result.code
                        if iteration < max_iterations:
                            console.print(f"  [yellow]↻ P{problem.problem_id} iter{iteration}: format error, retrying...[/yellow]")
                            continue
                        best_result = {
                            'problem_id': problem.problem_id,
                            'level': level,
                            'correct': False,
                            'compile_error': True,
                            'error_message': error_feedback,
                            'speedup': 0.0,
                        }
                        break

                    # Evaluate
                    result = runner.run_single(
                        formatted_code,
                        problem.pytorch_code,
                        problem.problem_id,
                        level
                    )

                    # Save generated code
                    code_path = output_path / f"problem_{problem.problem_id}.py"
                    with open(code_path, 'w') as f:
                        f.write(formatted_code)

                    if result['correct']:
                        # Success!
                        best_result = result
                        if iteration > 1:
                            console.print(f"  [green]✓ P{problem.problem_id}: fixed on iteration {iteration}[/green]")
                        break
                    else:
                        # Collect error for refinement
                        error_msg = result.get('error_message', 'Unknown error')
                        # Truncate very long tracebacks for the LLM prompt
                        if len(error_msg) > 1500:
                            error_msg = error_msg[:1500] + "\n... (truncated)"
                        error_feedback = error_msg
                        prev_code = formatted_code
                        best_result = result

                        if iteration < max_iterations:
                            error_type = "Compile" if result.get('compile_error') else "Runtime"
                            console.print(f"  [yellow]↻ P{problem.problem_id} iter{iteration}: {error_type} error, retrying...[/yellow]")
                        continue

                # Append best result for this problem
                results.append(best_result)

                # Brief status
                if best_result['correct']:
                    console.print(f"  [green]✓ P{problem.problem_id}: {best_result['speedup']:.2f}x speedup[/green]")
                else:
                    error_type = "Compile" if best_result.get('compile_error') else "Runtime"
                    console.print(f"  [red]✗ P{problem.problem_id}: {error_type} error (after {max_iterations} iterations)[/red]")

            progress.update(task, completed=len(problems))

        # Aggregate results
        summary = aggregator.aggregate(results, run_name=run_name, level=level)

        # Save results
        results_json = output_path / "results.json"
        aggregator.export_json(summary, str(results_json))

        results_csv = output_path / "results.csv"
        aggregator.export_csv(summary, str(results_csv))

        # Print summary
        console.print("\n")
        aggregator.print_summary(summary)

        console.print(f"\n[green]Results saved to: {output_path}[/green]")

    except ImportError as e:
        console.print(f"[red]Import error: {str(e)}[/red]")
        console.print("[yellow]Make sure all dependencies are installed: pip install -e .[/yellow]")
        raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    run_name: str = typer.Argument(..., help="Run name to analyze"),
    compare: Optional[str] = typer.Option(None, "--compare", "-c", help="Compare with another run"),
    export_csv: bool = typer.Option(False, "--csv", help="Export results as CSV"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed per-problem results"),
    output_dir: str = typer.Option("runs", "--output", "-o", help="Runs directory"),
):
    """Analyze benchmark results."""

    try:
        from llm_kernel.bench.results import ResultsAggregator, BenchmarkSummary

        aggregator = ResultsAggregator(console=console)

        # Load run results
        run_path = Path(output_dir) / run_name / "results.json"

        if not run_path.exists():
            console.print(f"[red]Run not found: {run_name}[/red]")
            console.print(f"[yellow]Expected: {run_path}[/yellow]")
            raise typer.Exit(1)

        with open(run_path, 'r') as f:
            data = json.load(f)

        summary = BenchmarkSummary(**data)

        # Print summary
        aggregator.print_summary(summary)

        if detailed:
            console.print("\n")
            aggregator.print_detailed_results(summary)

        # Compare if requested
        if compare:
            compare_path = Path(output_dir) / compare / "results.json"

            if not compare_path.exists():
                console.print(f"[red]Comparison run not found: {compare}[/red]")
                raise typer.Exit(1)

            with open(compare_path, 'r') as f:
                compare_data = json.load(f)

            compare_summary = BenchmarkSummary(**compare_data)

            comparison = aggregator.compare_runs(summary, compare_summary)

            console.print("\n")
            aggregator.print_comparison(comparison)

        # Export if requested
        if export_csv:
            csv_path = Path(output_dir) / run_name / "analysis.csv"
            aggregator.export_csv(summary, str(csv_path))

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


def check_vllm_installed() -> bool:
    """Check if vLLM is installed."""
    try:
        import vllm  # noqa
        return True
    except ImportError:
        return False


def check_server_health(url: str, timeout: float = 5.0) -> bool:
    """Check if an OpenAI-compatible server is healthy."""
    try:
        import httpx
        response = httpx.get(f"{url}/models", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


@app.command()
def server(
    action: str = typer.Argument(..., help="Action: start|stop|status|check"),
    model: str = typer.Option("qwen2.5-coder-32b", "--model", "-m", help="Model to serve"),
    gpu_util: float = typer.Option(0.7, "--gpu-util", help="GPU memory utilization (0.0-1.0)"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    tensor_parallel: int = typer.Option(1, "--tp", help="Tensor parallel size"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: vllm|ollama|external|auto"),
):
    """Manage LLM inference server (vLLM, Ollama, or external)."""

    action = action.lower()

    if action == "check":
        # Check available backends
        console.print("[cyan]Checking available backends...[/cyan]\n")

        vllm_ok = check_vllm_installed()
        console.print(f"  vLLM: {'[green]✓ installed[/green]' if vllm_ok else '[yellow]✗ not installed[/yellow]'}")

        # Check ollama
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
            ollama_ok = result.returncode == 0
        except Exception:
            ollama_ok = False
        console.print(f"  Ollama: {'[green]✓ installed[/green]' if ollama_ok else '[yellow]✗ not installed[/yellow]'}")

        # Check external server
        external_ok = check_server_health(f"http://localhost:{port}/v1")
        console.print(f"  External (port {port}): {'[green]✓ running[/green]' if external_ok else '[yellow]✗ not running[/yellow]'}")

        console.print("\n[cyan]Recommendations:[/cyan]")
        if not vllm_ok:
            console.print("  [yellow]vLLM requires Python <3.14. Current env is 3.14.[/yellow]")
            console.print("  [yellow]Options:[/yellow]")
            console.print("    1. Use Ollama: ollama serve && ollama pull qwen2.5-coder:32b")
            console.print("    2. Create Python 3.12 env: conda create -n llm_vllm python=3.12")
            console.print("    3. Use external server: --url http://your-server:8000/v1")
        return

    if action == "start":
        # Auto-detect backend
        if backend == "auto":
            if check_vllm_installed():
                backend = "vllm"
            else:
                console.print("[yellow]vLLM not installed (requires Python <3.14)[/yellow]")
                console.print("[cyan]Checking for Ollama...[/cyan]")
                try:
                    result = subprocess.run(["ollama", "--version"], capture_output=True, timeout=5)
                    if result.returncode == 0:
                        backend = "ollama"
                        console.print("[green]✓ Ollama found[/green]")
                    else:
                        backend = "external"
                except Exception:
                    backend = "external"

        if backend == "external":
            console.print("[red]No local backend available.[/red]")
            console.print("\n[cyan]Options to run LLM server:[/cyan]")
            console.print("  1. [yellow]Ollama (recommended):[/yellow]")
            console.print("     curl -fsSL https://ollama.com/install.sh | sh")
            console.print("     ollama serve")
            console.print("     ollama pull qwen2.5-coder:32b")
            console.print()
            console.print("  2. [yellow]vLLM (requires Python 3.12):[/yellow]")
            console.print("     conda create -n vllm_server python=3.12")
            console.print("     conda activate vllm_server")
            console.print("     pip install vllm")
            console.print("     vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000")
            console.print()
            console.print("  3. [yellow]Use external server:[/yellow]")
            console.print("     llm-kernel run --url http://your-server:8000/v1")
            raise typer.Exit(1)

        if backend == "ollama":
            # Use Ollama
            models_config = load_models_config()
            model_info = models_config.get(model, {})

            # Map HuggingFace name to Ollama name
            ollama_model_map = {
                "qwen2.5-coder-7b": "qwen2.5-coder:7b",
                "qwen2.5-coder-14b": "qwen2.5-coder:14b",
                "qwen2.5-coder-32b": "qwen2.5-coder:32b",
                "deepseek-coder-v2-lite": "deepseek-coder-v2:16b",
                "codestral-22b": "codestral:22b",
            }
            ollama_model = ollama_model_map.get(model, model)

            console.print(f"[cyan]Starting Ollama with {ollama_model}...[/cyan]")
            console.print(f"[yellow]Make sure ollama is serving: ollama serve[/yellow]")
            console.print(f"[yellow]Pull model if needed: ollama pull {ollama_model}[/yellow]")
            console.print()
            console.print(f"[green]Ollama OpenAI endpoint: http://localhost:11434/v1[/green]")
            console.print(f"[cyan]Use with: llm-kernel run --url http://localhost:11434/v1 --model {ollama_model}[/cyan]")
            return

        # vLLM backend
        pid = find_vllm_process()
        if pid:
            console.print(f"[yellow]vLLM server already running (PID: {pid})[/yellow]")
            return

        models_config = load_models_config()

        if model not in models_config:
            console.print(f"[red]Model not found in config: {model}[/red]")
            console.print("[yellow]Available models: run 'llm-kernel models' to list[/yellow]")
            raise typer.Exit(1)

        model_info = models_config[model]
        model_name = model_info['name']

        console.print(f"[cyan]Starting vLLM server for {model_name}...[/cyan]")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_util),
            "--tensor-parallel-size", str(tensor_parallel),
        ]

        if model_info.get('dtype'):
            cmd.extend(["--dtype", model_info['dtype']])

        if model_info.get('quantization'):
            cmd.extend(["--quantization", model_info['quantization']])

        console.print(f"[yellow]Command: {' '.join(cmd)}[/yellow]")

        log_path = get_project_root() / "vllm_server.log"
        with open(log_path, 'w') as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        console.print(f"[green]✓ Server starting (PID: {process.pid})[/green]")
        console.print(f"[yellow]Logs: {log_path}[/yellow]")
        console.print(f"[cyan]Check status with: llm-kernel server status[/cyan]")

    elif action == "stop":
        pid = find_vllm_process()

        if not pid:
            console.print("[yellow]No vLLM server running[/yellow]")
            return

        console.print(f"[cyan]Stopping vLLM server (PID: {pid})...[/cyan]")

        try:
            subprocess.run(["kill", str(pid)], check=True)
            console.print("[green]✓ Server stopped[/green]")
        except subprocess.CalledProcessError:
            console.print("[red]Failed to stop server[/red]")
            console.print(f"[yellow]Try: kill {pid}[/yellow]")
            raise typer.Exit(1)

    elif action == "status":
        pid = find_vllm_process()

        if pid:
            console.print(f"[green]✓ vLLM server running (PID: {pid})[/green]")

            # Try to get model info
            try:
                import requests
                response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
                if response.ok:
                    models = response.json().get('data', [])
                    if models:
                        console.print(f"[cyan]Model: {models[0]['id']}[/cyan]")
            except Exception:
                pass

        else:
            console.print("[yellow]vLLM server not running[/yellow]")
            console.print("[cyan]Start with: llm-kernel server start[/cyan]")

    else:
        console.print(f"[red]Invalid action: {action}[/red]")
        console.print("[yellow]Valid actions: start, stop, status[/yellow]")
        raise typer.Exit(1)


@app.command()
def models(
    show_vram: bool = typer.Option(True, "--vram", help="Show VRAM estimates"),
    filter_fits: bool = typer.Option(False, "--fits", help="Only show models that fit in 96GB"),
):
    """List available models with VRAM estimates."""

    models_config = load_models_config()

    if not models_config:
        console.print("[red]No models configuration found[/red]")
        raise typer.Exit(1)

    table = Table(title="Available Models")

    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Model Name", style="magenta")
    table.add_column("Params", style="blue")

    if show_vram:
        table.add_column("VRAM (FP16)", style="yellow")
        table.add_column("Strategy", style="green")
        table.add_column("GPU Util", style="green")

    table.add_column("Strengths", style="white")

    for alias, info in sorted(models_config.items()):
        vram_gb = info.get('vram_fp16_gb')
        vram_strategy = info.get('vram_strategy', 'original')

        # Filter if requested
        if filter_fits and vram_gb and vram_gb > 76:
            continue

        row = [
            alias,
            info['name'],
            info.get('params', 'Unknown'),
        ]

        if show_vram:
            vram_str = f"{vram_gb} GB" if vram_gb else "Unknown"
            gpu_util = info.get('gpu_memory_utilization', 0.5)

            row.extend([
                vram_str,
                vram_strategy,
                f"{gpu_util:.1%}",
            ])

        # Format strengths
        strengths = info.get('strengths', [])
        strengths_str = ", ".join(strengths[:2]) if strengths else "General purpose"

        row.append(strengths_str)

        table.add_row(*row)

    console.print(table)

    # Show VRAM budget info
    vram_config = load_models_config()
    if 'vram_strategy' in vram_config:
        budget = vram_config['vram_strategy']
        console.print(f"\n[cyan]VRAM Budget: {budget.get('total_vram_gb')} GB total, "
                     f"{budget.get('max_model_vram_gb')} GB for model[/cyan]")


@app.command()
def tritonbench(
    action: str = typer.Argument(..., help="Action: 'list' or 'run'"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Problem name to run"),
    channel: str = typer.Option("T", "--channel", "-c", help="Channel: 'T' or 'G'"),
    stats: bool = typer.Option(False, "--stats", help="Show statistics"),
):
    """
    TritonBench integration commands.

    Examples:
        llm-kernel tritonbench list
        llm-kernel tritonbench list --channel G --stats
        llm-kernel tritonbench run --name matmul
    """
    from llm_kernel.bench.tritonbench_runner import TritonBenchRunner

    try:
        # Initialize runner with relative path from project root
        project_root = get_project_root()
        repo_path = project_root / "external" / "TritonBench"

        if not repo_path.exists():
            console.print(f"[red]TritonBench not found at {repo_path}[/red]")
            console.print("[yellow]Run: git clone https://github.com/thunlp/TritonBench.git external/TritonBench[/yellow]")
            raise typer.Exit(1)

        runner = TritonBenchRunner(repo_path=str(repo_path))

        if action == "list":
            # List problems
            problems = runner.list_problems(channel)

            if not problems:
                console.print(f"[yellow]No problems found in channel {channel}[/yellow]")
                return

            # Create table
            table = Table(title=f"TritonBench-{channel} Problems ({len(problems)} total)")
            table.add_column("Name", style="cyan")
            table.add_column("Difficulty", style="yellow")
            table.add_column("Params", style="green")
            table.add_column("Torch Ops", style="magenta")
            table.add_column("File", style="dim")

            for p in problems:
                table.add_row(
                    p.name,
                    str(p.difficulty),
                    str(p.params_cnt),
                    str(p.torch_cnt),
                    p.file
                )

            console.print(table)

            # Show statistics if requested
            if stats:
                summary = runner.get_summary_stats(channel)
                console.print("\n[bold cyan]Statistics:[/bold cyan]")
                console.print(f"  Total problems: {summary['total']}")
                console.print(f"  Avg difficulty: {summary['avg_difficulty']:.2f}")
                console.print(f"  Avg params: {summary['avg_params']:.1f}")
                console.print(f"  Avg torch ops: {summary['avg_torch_ops']:.1f}")
                console.print(f"  Difficulty range: {summary['difficulty_range']}")

        elif action == "run":
            if not name:
                console.print("[red]Error: --name is required for 'run' action[/red]")
                console.print("Example: llm-kernel tritonbench run --name matmul")
                raise typer.Exit(1)

            # Get problem
            problem = runner.get_problem_by_name(name, channel)

            if not problem:
                console.print(f"[red]Problem '{name}' not found in channel {channel}[/red]")
                raise typer.Exit(1)

            # Read problem code
            code = runner.read_problem_code(problem, channel)

            # Display problem info
            panel = Panel(
                f"[bold]Description:[/bold]\n{problem.description}\n\n"
                f"[bold]Difficulty:[/bold] {problem.difficulty}\n"
                f"[bold]Parameters:[/bold] {problem.params_cnt}\n"
                f"[bold]Torch Operations:[/bold] {problem.torch_cnt}\n\n"
                f"[bold]Reference Code:[/bold]\n{code[:500]}...",
                title=f"TritonBench-{channel}: {problem.name}",
                expand=False
            )
            console.print(panel)

            console.print("\n[yellow]Note: Full pipeline integration with LLM generation coming soon[/yellow]")

        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("[yellow]Valid actions: 'list', 'run'[/yellow]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
