"""
KernelBench Results Aggregation and Analysis.

Aggregates benchmark results, computes metrics, and provides
comparison and export utilities.
"""

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""
    run_name: str
    level: Optional[int]  # None means all levels
    total_problems: int
    correct_count: int
    fast_0: float  # Correctness rate
    fast_1: float  # Correct AND faster than baseline (speedup > 1.0)
    fast_2: float  # Correct AND 2x faster (speedup > 2.0)
    avg_speedup: float
    median_speedup: float
    per_problem: List[Dict[str, Any]]
    timestamp: str = None

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ResultsAggregator:
    """
    Aggregate and analyze KernelBench evaluation results.

    Computes standard KernelBench metrics:
    - Fast-0: Correctness rate (correct / total)
    - Fast-1: Speedup >= 1.0x rate
    - Fast-2: Speedup >= 2.0x rate
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize results aggregator.

        Args:
            console: Rich Console instance for output (creates new if None)
        """
        self.console = console or Console()

    def aggregate(
        self,
        results: List[Dict[str, Any]],
        run_name: str = "default",
        level: Optional[int] = None
    ) -> BenchmarkSummary:
        """
        Aggregate results into summary statistics.

        Args:
            results: List of result dictionaries from BenchRunner
            run_name: Name for this benchmark run
            level: Specific level being summarized (None for all levels)

        Returns:
            BenchmarkSummary with aggregated metrics
        """
        if not results:
            return BenchmarkSummary(
                run_name=run_name,
                level=level,
                total_problems=0,
                correct_count=0,
                fast_0=0.0,
                fast_1=0.0,
                fast_2=0.0,
                avg_speedup=0.0,
                median_speedup=0.0,
                per_problem=[],
            )

        total_problems = len(results)
        correct_results = [r for r in results if r.get('correct', False)]
        correct_count = len(correct_results)

        # Fast-0: Correctness rate
        fast_0 = correct_count / total_problems if total_problems > 0 else 0.0

        # Fast-1: Speedup >= 1.0x
        fast_1_count = sum(
            1 for r in correct_results if r.get('speedup', 0.0) >= 1.0
        )
        fast_1 = fast_1_count / total_problems if total_problems > 0 else 0.0

        # Fast-2: Speedup >= 2.0x
        fast_2_count = sum(
            1 for r in correct_results if r.get('speedup', 0.0) >= 2.0
        )
        fast_2 = fast_2_count / total_problems if total_problems > 0 else 0.0

        # Speedup statistics (only for correct results)
        speedups = [r.get('speedup', 0.0) for r in correct_results]

        if speedups:
            avg_speedup = np.mean(speedups)
            median_speedup = np.median(speedups)
        else:
            avg_speedup = 0.0
            median_speedup = 0.0

        return BenchmarkSummary(
            run_name=run_name,
            level=level,
            total_problems=total_problems,
            correct_count=correct_count,
            fast_0=fast_0,
            fast_1=fast_1,
            fast_2=fast_2,
            avg_speedup=avg_speedup,
            median_speedup=median_speedup,
            per_problem=results,
        )

    def compare_runs(
        self,
        run1: BenchmarkSummary,
        run2: BenchmarkSummary
    ) -> Dict[str, Any]:
        """
        Compare two benchmark runs.

        Args:
            run1: First benchmark summary
            run2: Second benchmark summary

        Returns:
            Dictionary with comparison metrics
        """
        comparison = {
            'run1_name': run1.run_name,
            'run2_name': run2.run_name,
            'delta_fast_0': run2.fast_0 - run1.fast_0,
            'delta_fast_1': run2.fast_1 - run1.fast_1,
            'delta_fast_2': run2.fast_2 - run1.fast_2,
            'delta_avg_speedup': run2.avg_speedup - run1.avg_speedup,
            'delta_median_speedup': run2.median_speedup - run1.median_speedup,
            'delta_correct_count': run2.correct_count - run1.correct_count,
        }

        # Per-problem comparison (if same problems)
        if run1.total_problems == run2.total_problems:
            comparison['same_problem_count'] = True

            # Find problems that changed correctness
            improved = []
            regressed = []

            for p1, p2 in zip(run1.per_problem, run2.per_problem):
                pid = p1.get('problem_id')
                if p1.get('correct') and not p2.get('correct'):
                    regressed.append(pid)
                elif not p1.get('correct') and p2.get('correct'):
                    improved.append(pid)

            comparison['improved_problems'] = improved
            comparison['regressed_problems'] = regressed

        else:
            comparison['same_problem_count'] = False

        return comparison

    def export_csv(self, summary: BenchmarkSummary, path: str) -> None:
        """
        Export summary results to CSV file.

        Args:
            summary: Benchmark summary to export
            path: Output CSV file path
        """
        csv_path = Path(path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'problem_id',
                'level',
                'correct',
                'speedup',
                'triton_time_ms',
                'reference_time_ms',
                'compile_error',
                'runtime_error',
                'error_message'
            ])

            # Write per-problem results
            for result in summary.per_problem:
                writer.writerow([
                    result.get('problem_id', ''),
                    result.get('level', ''),
                    result.get('correct', False),
                    f"{result.get('speedup', 0.0):.3f}",
                    f"{result.get('triton_time_ms', 0.0):.3f}",
                    f"{result.get('reference_time_ms', 0.0):.3f}",
                    result.get('compile_error', False),
                    result.get('runtime_error', False),
                    result.get('error_message', ''),
                ])

        self.console.print(f"[green]Results exported to {csv_path}[/green]")

    def export_json(self, summary: BenchmarkSummary, path: str) -> None:
        """
        Export summary results to JSON file.

        Args:
            summary: Benchmark summary to export
            path: Output JSON file path
        """
        json_path = Path(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        self.console.print(f"[green]Results exported to {json_path}[/green]")

    def print_summary(self, summary: BenchmarkSummary) -> None:
        """
        Print formatted summary to console.

        Args:
            summary: Benchmark summary to display
        """
        # Create summary table
        table = Table(title=f"Benchmark Summary: {summary.run_name}")

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        # Add rows
        level_str = f"Level {summary.level}" if summary.level else "All Levels"
        table.add_row("Level", level_str)
        table.add_row("Total Problems", str(summary.total_problems))
        table.add_row("Correct", f"{summary.correct_count} ({summary.fast_0:.1%})")
        table.add_row("Fast-0 (Correctness)", f"{summary.fast_0:.1%}")
        table.add_row("Fast-1 (Speedup ≥ 1.0x)", f"{summary.fast_1:.1%}")
        table.add_row("Fast-2 (Speedup ≥ 2.0x)", f"{summary.fast_2:.1%}")
        table.add_row("Avg Speedup", f"{summary.avg_speedup:.2f}x")
        table.add_row("Median Speedup", f"{summary.median_speedup:.2f}x")

        self.console.print(table)

        # Error breakdown
        compile_errors = sum(1 for r in summary.per_problem if r.get('compile_error'))
        runtime_errors = sum(1 for r in summary.per_problem if r.get('runtime_error'))

        if compile_errors > 0 or runtime_errors > 0:
            error_table = Table(title="Error Breakdown")
            error_table.add_column("Error Type", style="red")
            error_table.add_column("Count", style="red")

            error_table.add_row("Compile Errors", str(compile_errors))
            error_table.add_row("Runtime Errors", str(runtime_errors))

            self.console.print(error_table)

    def print_detailed_results(self, summary: BenchmarkSummary, top_n: int = 10) -> None:
        """
        Print detailed per-problem results.

        Args:
            summary: Benchmark summary
            top_n: Number of top/bottom problems to show
        """
        # Sort by speedup
        sorted_results = sorted(
            [r for r in summary.per_problem if r.get('correct')],
            key=lambda x: x.get('speedup', 0.0),
            reverse=True
        )

        if sorted_results:
            # Top performers
            top_table = Table(title=f"Top {top_n} Best Speedups")
            top_table.add_column("Problem", style="cyan")
            top_table.add_column("Level", style="blue")
            top_table.add_column("Speedup", style="green")
            top_table.add_column("Triton (ms)", style="yellow")
            top_table.add_column("Ref (ms)", style="yellow")

            for result in sorted_results[:top_n]:
                top_table.add_row(
                    f"P{result.get('problem_id', '')}",
                    str(result.get('level', '')),
                    f"{result.get('speedup', 0.0):.2f}x",
                    f"{result.get('triton_time_ms', 0.0):.3f}",
                    f"{result.get('reference_time_ms', 0.0):.3f}",
                )

            self.console.print(top_table)

        # Failed problems
        failed_results = [r for r in summary.per_problem if not r.get('correct')]

        if failed_results:
            fail_table = Table(title="Failed Problems")
            fail_table.add_column("Problem", style="cyan")
            fail_table.add_column("Level", style="blue")
            fail_table.add_column("Error Type", style="red")
            fail_table.add_column("Message", style="red")

            for result in failed_results[:top_n]:
                error_type = "Compile" if result.get('compile_error') else "Runtime"
                error_msg = result.get('error_message', '')[:50]  # Truncate

                fail_table.add_row(
                    f"P{result.get('problem_id', '')}",
                    str(result.get('level', '')),
                    error_type,
                    error_msg,
                )

            self.console.print(fail_table)

    def print_comparison(self, comparison: Dict[str, Any]) -> None:
        """
        Print comparison between two runs.

        Args:
            comparison: Comparison dictionary from compare_runs()
        """
        table = Table(title=f"Comparison: {comparison['run1_name']} vs {comparison['run2_name']}")

        table.add_column("Metric", style="cyan")
        table.add_column("Delta", style="magenta")
        table.add_column("Direction", style="green")

        metrics = [
            ("Fast-0 (Correctness)", comparison['delta_fast_0'], "%"),
            ("Fast-1 (Speedup ≥ 1.0x)", comparison['delta_fast_1'], "%"),
            ("Fast-2 (Speedup ≥ 2.0x)", comparison['delta_fast_2'], "%"),
            ("Avg Speedup", comparison['delta_avg_speedup'], "x"),
            ("Median Speedup", comparison['delta_median_speedup'], "x"),
            ("Correct Count", comparison['delta_correct_count'], ""),
        ]

        for metric_name, delta, unit in metrics:
            if unit == "%":
                delta_str = f"{delta:+.1%}"
            elif unit == "x":
                delta_str = f"{delta:+.2f}x"
            else:
                delta_str = f"{delta:+d}"

            direction = "↑" if delta > 0 else "↓" if delta < 0 else "="
            table.add_row(metric_name, delta_str, direction)

        self.console.print(table)

        # Show improved/regressed problems
        if comparison.get('same_problem_count'):
            improved = comparison.get('improved_problems', [])
            regressed = comparison.get('regressed_problems', [])

            if improved or regressed:
                change_table = Table(title="Problem Changes")
                change_table.add_column("Type", style="cyan")
                change_table.add_column("Count", style="magenta")
                change_table.add_column("Problem IDs", style="yellow")

                if improved:
                    change_table.add_row(
                        "Improved",
                        str(len(improved)),
                        ", ".join(f"P{pid}" for pid in improved[:10])
                    )

                if regressed:
                    change_table.add_row(
                        "Regressed",
                        str(len(regressed)),
                        ", ".join(f"P{pid}" for pid in regressed[:10])
                    )

                self.console.print(change_table)

    def get_level_breakdown(self, summary: BenchmarkSummary) -> Dict[int, Dict[str, float]]:
        """
        Get per-level breakdown of metrics.

        Args:
            summary: Benchmark summary

        Returns:
            Dictionary mapping level -> metrics
        """
        level_results = {}

        for level in range(1, 5):
            level_problems = [r for r in summary.per_problem if r.get('level') == level]

            if not level_problems:
                continue

            level_summary = self.aggregate(level_problems, f"{summary.run_name}_L{level}", level)

            level_results[level] = {
                'total': level_summary.total_problems,
                'correct': level_summary.correct_count,
                'fast_0': level_summary.fast_0,
                'fast_1': level_summary.fast_1,
                'fast_2': level_summary.fast_2,
                'avg_speedup': level_summary.avg_speedup,
                'median_speedup': level_summary.median_speedup,
            }

        return level_results
