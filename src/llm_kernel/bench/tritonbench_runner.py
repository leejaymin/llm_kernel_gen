"""
TritonBench Runner and Integration.

Provides wrapper classes to run TritonBench benchmarks
and convert results to KernelBench-compatible format.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

from rich.console import Console

console = Console()


@dataclass
class TritonBenchProblem:
    """TritonBench problem representation."""
    name: str
    func_inputs: str
    description: str
    math: str
    example: str
    torch_code: str
    torch_cnt: int
    difficulty: int
    params_cnt: int
    file: str
    other: Optional[str] = None


class TritonBenchRunner:
    """Runner for TritonBench benchmarks."""

    def __init__(self, repo_path: str = "external/TritonBench"):
        """Initialize TritonBench runner.

        Args:
            repo_path: Path to TritonBench repository
        """
        self.repo_path = Path(repo_path)
        if not self.repo_path.exists():
            raise FileNotFoundError(f"TritonBench not found at {repo_path}")

        self.data_path = self.repo_path / "data"
        self.eval_path = self.repo_path / "EVAL"

        # TritonBench channels
        self.tritonbench_t_path = self.data_path / "TritonBench_T_v1"
        self.tritonbench_g_path = self.data_path / "TritonBench_G_v1"

        # Metadata files
        self.t_metadata = self.data_path / "TritonBench_T_v1.jsonl"
        self.g_metadata = self.data_path / "TritonBench_G_v1.json"

    def list_problems(self, channel: str = "T") -> List[TritonBenchProblem]:
        """List all problems in a channel.

        Args:
            channel: "T" for TritonBench-T or "G" for TritonBench-G

        Returns:
            List of TritonBenchProblem objects
        """
        if channel == "T":
            metadata_file = self.t_metadata
        elif channel == "G":
            metadata_file = self.g_metadata
        else:
            raise ValueError(f"Invalid channel: {channel}. Use 'T' or 'G'")

        if not metadata_file.exists():
            console.print(f"[yellow]Warning: {metadata_file} not found[/yellow]")
            return []

        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        problems = []
        for item in data:
            if channel == "G":
                # Channel G has different format (no name, uses file without .py)
                name = item["file"].replace(".py", "")
                problem = TritonBenchProblem(
                    name=name,
                    func_inputs=item.get("simp_instru", ""),
                    description=item.get("comp_instru", ""),
                    math="",
                    example="",
                    torch_code=item.get("output", ""),
                    torch_cnt=0,
                    difficulty=int(item.get("difficulty", 0)),
                    params_cnt=0,
                    file=item["file"],
                    other=item.get("repo")
                )
            else:
                # Channel T has standard format
                problem = TritonBenchProblem(
                    name=item["name"],
                    func_inputs=item["func_inputs"],
                    description=item["description"],
                    math=item.get("math", ""),
                    example=item.get("example", ""),
                    torch_code=item.get("torch_code", ""),
                    torch_cnt=item.get("torch_cnt", 0),
                    difficulty=item.get("difficulty", 0),
                    params_cnt=item.get("params_cnt", 0),
                    file=item["file"],
                    other=item.get("other")
                )
            problems.append(problem)

        return problems

    def get_problem_by_name(self, name: str, channel: str = "T") -> Optional[TritonBenchProblem]:
        """Get a specific problem by name.

        Args:
            name: Problem name
            channel: "T" or "G"

        Returns:
            TritonBenchProblem or None if not found
        """
        problems = self.list_problems(channel)
        for p in problems:
            if p.name == name:
                return p
        return None

    def get_problem_file(self, problem: TritonBenchProblem, channel: str = "T") -> Path:
        """Get the file path for a problem.

        Args:
            problem: TritonBenchProblem object
            channel: "T" or "G"

        Returns:
            Path to the problem file
        """
        if channel == "T":
            return self.tritonbench_t_path / problem.file
        else:
            return self.tritonbench_g_path / problem.file

    def read_problem_code(self, problem: TritonBenchProblem, channel: str = "T") -> str:
        """Read the reference PyTorch code for a problem.

        Args:
            problem: TritonBenchProblem object
            channel: "T" or "G"

        Returns:
            Python code as string
        """
        file_path = self.get_problem_file(problem, channel)
        if not file_path.exists():
            raise FileNotFoundError(f"Problem file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def run_call_accuracy(
        self,
        source_path: str,
        target_path: str,
        channel: str = "T",
        gpus: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Run call accuracy evaluation (0_call_acc.py).

        Args:
            source_path: Path to reference implementations
            target_path: Path to generated implementations
            channel: "T" or "G"
            gpus: List of GPU IDs to use

        Returns:
            Evaluation results
        """
        eval_script = self.eval_path / f"eval_{channel}" / "0_call_acc.py"
        if not eval_script.exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

        cmd = [
            "python",
            str(eval_script),
            "--source", source_path,
            "--target", target_path
        ]

        if gpus:
            cmd.extend(["--GPUs", str(gpus)])

        console.print(f"[cyan]Running call accuracy: {' '.join(cmd)}[/cyan]")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=300
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Timeout after 300 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def run_execution_accuracy(
        self,
        folder: str,
        channel: str = "T",
        gpus: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Run execution accuracy evaluation (1_exe_acc.py).

        Args:
            folder: Path to generated implementations
            channel: "T" or "G"
            gpus: List of GPU IDs to use

        Returns:
            Evaluation results
        """
        eval_script = self.eval_path / f"eval_{channel}" / "1_exe_acc.py"
        if not eval_script.exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_script}")

        cmd = [
            "python",
            str(eval_script),
            "--folder", folder
        ]

        if gpus:
            cmd.extend(["--GPUs", str(gpus)])

        console.print(f"[cyan]Running execution accuracy: {' '.join(cmd)}[/cyan]")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=600
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Timeout after 600 seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_summary_stats(self, channel: str = "T") -> Dict[str, Any]:
        """Get summary statistics for a channel.

        Args:
            channel: "T" or "G"

        Returns:
            Summary statistics
        """
        problems = self.list_problems(channel)

        if not problems:
            return {
                "total": 0,
                "channel": channel
            }

        difficulties = [p.difficulty for p in problems]
        param_counts = [p.params_cnt for p in problems]
        torch_counts = [p.torch_cnt for p in problems]

        return {
            "channel": channel,
            "total": len(problems),
            "avg_difficulty": sum(difficulties) / len(difficulties) if difficulties else 0,
            "avg_params": sum(param_counts) / len(param_counts) if param_counts else 0,
            "avg_torch_ops": sum(torch_counts) / len(torch_counts) if torch_counts else 0,
            "difficulty_range": (min(difficulties), max(difficulties)) if difficulties else (0, 0),
            "problems": [p.name for p in problems]
        }
