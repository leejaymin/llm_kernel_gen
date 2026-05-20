#!/usr/bin/env python3
"""B. Iteration ablation study: test 1/3/10 iterations with gpt-4o on Level 1.
We already have 5-iteration results from gpt4o_level1_v2.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "gpt-4o"
LEVEL = 1
NUM_PROBLEMS = 100

for n_iter in [1, 3, 10]:
    run_prefix = f"ablation_iter{n_iter}"
    output_dir = Path(f"runs/{run_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    solved = 0

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
        print(f"Resuming iter={n_iter}: {len(already_done)} done, {solved} solved")

    print(f"\n{'#'*60}")
    print(f"  Ablation: {MODEL}, Level {LEVEL}, iterations={n_iter}")
    print(f"{'#'*60}")

    for i, pid in enumerate(range(1, NUM_PROBLEMS + 1)):
        if pid in already_done:
            continue

        print(f"\n[{i+1}/{NUM_PROBLEMS}] iter={n_iter}, P{pid}...", end=" ", flush=True)
        start = time.time()
        individual_name = f"{run_prefix}_p{pid}"

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "llm_kernel.cli", "run",
                    "--level", str(LEVEL),
                    "--problem", str(pid),
                    "--provider", "openai",
                    "--model", MODEL,
                    "--name", individual_name,
                    "--iterations", str(n_iter),
                ],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
                cwd=str(Path(__file__).parent),
            )

            elapsed = time.time() - start
            individual_result_path = Path(f"runs/{individual_name}/results.json")
            if individual_result_path.exists():
                with open(individual_result_path) as f:
                    prob_data = json.load(f)
                if prob_data.get("per_problem"):
                    r = prob_data["per_problem"][0]
                    all_results.append(r)
                    if r["correct"]:
                        solved += 1
                        print(f"SOLVED ({r.get('speedup',0):.2f}x) in {elapsed:.0f}s")
                    else:
                        print(f"failed in {elapsed:.0f}s")
                else:
                    all_results.append({
                        "problem_id": pid, "level": LEVEL, "correct": False,
                        "speedup": 0.0, "compile_error": True, "runtime_error": False,
                        "error_message": "No results"
                    })
                    print(f"no results in {elapsed:.0f}s")
            else:
                all_results.append({
                    "problem_id": pid, "level": LEVEL, "correct": False,
                    "speedup": 0.0, "compile_error": True, "runtime_error": False,
                    "error_message": "Results file not found"
                })
                print(f"no file in {elapsed:.0f}s")

        except subprocess.TimeoutExpired:
            print(f"TIMEOUT")
            all_results.append({
                "problem_id": pid, "level": LEVEL, "correct": False,
                "speedup": 0.0, "compile_error": False, "runtime_error": True,
                "error_message": "Timeout (300s)"
            })

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "problem_id": pid, "level": LEVEL, "correct": False,
                "speedup": 0.0, "compile_error": True, "runtime_error": False,
                "error_message": str(e)
            })

        # Save partial
        partial = {
            "run_name": run_prefix,
            "model": MODEL,
            "level": LEVEL,
            "iterations": n_iter,
            "total_problems": NUM_PROBLEMS,
            "solved_so_far": solved,
            "per_problem": all_results,
        }
        with open(results_file, 'w') as f:
            json.dump(partial, f, indent=2)

    # Save final
    final = {
        "run_name": run_prefix,
        "model": MODEL,
        "level": LEVEL,
        "iterations": n_iter,
        "total_problems": NUM_PROBLEMS,
        "solved_count": solved,
        "per_problem": all_results,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n>>> iter={n_iter}: {solved}/{NUM_PROBLEMS} solved ({solved/NUM_PROBLEMS*100:.1f}%)")

# Summary
print(f"\n{'='*60}")
print("ABLATION SUMMARY")
print(f"{'='*60}")
for n_iter in [1, 3, 5, 10]:
    if n_iter == 5:
        f = Path("runs/gpt4o_level1_v2/results.json")
    else:
        f = Path(f"runs/ablation_iter{n_iter}/results.json")
    if f.exists():
        d = json.load(open(f))
        solved = sum(1 for r in d["per_problem"] if r["correct"])
        print(f"  iterations={n_iter:>2d}: {solved}/100 ({solved}%)")
