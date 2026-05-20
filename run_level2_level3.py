#!/usr/bin/env python3
"""Run KernelBench Level 2 and Level 3 experiments.

Phase 1: gpt-4o baseline on all problems
Phase 2: gpt-5.2 on gpt-4o failed problems
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_model_on_problems(
    model: str,
    level: int,
    problem_ids: list,
    run_prefix: str,
    iterations: int = 5,
    timeout: int = 300,
):
    """Run a model on a list of problems, with resume support."""
    output_dir = Path(f"runs/{run_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    solved = 0
    total = len(problem_ids)

    # Resume support
    already_done = set()
    results_file = output_dir / "results_partial.json"
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)
        for r in existing.get("per_problem", []):
            already_done.add(r["problem_id"])
            all_results.append(r)
            if r["correct"]:
                solved += 1
        print(f"Resuming: {len(already_done)} done, {solved} solved")

    for i, pid in enumerate(problem_ids):
        if pid in already_done:
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Level {level} Problem {pid} with {model}...")
        print(f"{'='*60}")

        start = time.time()
        individual_name = f"{run_prefix}_p{pid}"

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "llm_kernel.cli", "run",
                    "--level", str(level),
                    "--problem", str(pid),
                    "--provider", "openai",
                    "--model", model,
                    "--name", individual_name,
                    "--iterations", str(iterations),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
                cwd=str(Path(__file__).parent),
            )

            elapsed = time.time() - start
            print(f"Completed in {elapsed:.1f}s")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

            if result.stderr:
                stderr_lines = [l for l in result.stderr.split('\n')
                              if 'ResourceTracker' not in l and '_recursion_count' not in l and l.strip()]
                if stderr_lines:
                    print("STDERR:", '\n'.join(stderr_lines[-3:]))

            # Read individual result
            individual_result_path = Path(f"runs/{individual_name}/results.json")
            if individual_result_path.exists():
                with open(individual_result_path) as f:
                    prob_data = json.load(f)
                if prob_data.get("per_problem"):
                    r = prob_data["per_problem"][0]
                    all_results.append(r)
                    if r["correct"]:
                        solved += 1
                        print(f"  >>> SOLVED! Speedup: {r.get('speedup', 0):.2f}x")
                    else:
                        print(f"  >>> Failed: compile={r.get('compile_error')}, runtime={r.get('runtime_error')}")
                else:
                    all_results.append({
                        "problem_id": pid, "level": level, "correct": False,
                        "speedup": 0.0, "compile_error": True, "runtime_error": False,
                        "error_message": "No results"
                    })
            else:
                all_results.append({
                    "problem_id": pid, "level": level, "correct": False,
                    "speedup": 0.0, "compile_error": True, "runtime_error": False,
                    "error_message": "Results file not found"
                })

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"  TIMEOUT after {elapsed:.1f}s")
            all_results.append({
                "problem_id": pid, "level": level, "correct": False,
                "speedup": 0.0, "compile_error": False, "runtime_error": True,
                "error_message": f"Timeout ({timeout}s)"
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "problem_id": pid, "level": level, "correct": False,
                "speedup": 0.0, "compile_error": True, "runtime_error": False,
                "error_message": str(e)
            })

        # Save partial results
        partial = {
            "run_name": run_prefix,
            "model": model,
            "level": level,
            "total_problems": total,
            "solved_so_far": solved,
            "per_problem": all_results,
        }
        with open(results_file, 'w') as f:
            json.dump(partial, f, indent=2)

        print(f"\nProgress: {solved}/{len(all_results)} solved ({solved/max(len(all_results),1)*100:.1f}%)")

    # Save final
    final = {
        "run_name": run_prefix,
        "model": model,
        "level": level,
        "total_problems": total,
        "solved_count": solved,
        "per_problem": all_results,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(final, f, indent=2)

    return final


def run_level(level: int, num_problems: int):
    """Run full pipeline for a level: gpt-4o baseline, then gpt-5.2 on failures."""
    all_problems = list(range(1, num_problems + 1))

    print(f"\n{'#'*70}")
    print(f"  PHASE 1: gpt-4o baseline on Level {level} ({num_problems} problems)")
    print(f"{'#'*70}")

    gpt4o_results = run_model_on_problems(
        model="gpt-4o",
        level=level,
        problem_ids=all_problems,
        run_prefix=f"gpt4o_level{level}",
        iterations=5,
        timeout=300,
    )

    gpt4o_solved_ids = set(
        r["problem_id"] for r in gpt4o_results["per_problem"] if r["correct"]
    )
    gpt4o_solved_count = len(gpt4o_solved_ids)
    failed_problems = sorted(set(all_problems) - gpt4o_solved_ids)

    print(f"\n{'#'*70}")
    print(f"  gpt-4o Level {level} Results: {gpt4o_solved_count}/{num_problems}")
    print(f"  Failed: {len(failed_problems)} problems")
    print(f"{'#'*70}")

    if not failed_problems:
        print("All problems solved! No need to run gpt-5.2.")
        return

    print(f"\n{'#'*70}")
    print(f"  PHASE 2: gpt-5.2 on {len(failed_problems)} failed problems (Level {level})")
    print(f"{'#'*70}")

    gpt52_results = run_model_on_problems(
        model="gpt-5.2",
        level=level,
        problem_ids=failed_problems,
        run_prefix=f"gpt52_level{level}_failed",
        iterations=5,
        timeout=300,
    )

    gpt52_solved_ids = set(
        r["problem_id"] for r in gpt52_results["per_problem"] if r["correct"]
    )
    combined = gpt4o_solved_ids | gpt52_solved_ids

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS - Level {level}")
    print(f"{'='*70}")
    print(f"  gpt-4o:                 {gpt4o_solved_count}/{num_problems} ({gpt4o_solved_count/num_problems*100:.1f}%)")
    print(f"  gpt-5.2 on failures:    {len(gpt52_solved_ids)}/{len(failed_problems)} ({len(gpt52_solved_ids)/max(len(failed_problems),1)*100:.1f}%)")
    print(f"  Combined:               {len(combined)}/{num_problems} ({len(combined)/num_problems*100:.1f}%)")
    print(f"  gpt-5.2 solved: {sorted(gpt52_solved_ids)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True, choices=[2, 3])
    args = parser.parse_args()

    if args.level == 2:
        run_level(2, 100)
    elif args.level == 3:
        run_level(3, 50)
