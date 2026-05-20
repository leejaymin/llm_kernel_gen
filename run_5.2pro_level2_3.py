#!/usr/bin/env python3
"""A. gpt-5.2-pro on Level 2/3 failed problems (gpt-4o failures)."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL = "gpt-5.2-pro"

for level, num_problems in [(2, 100), (3, 50)]:
    # Load gpt-4o results to find failed problems
    f4o = Path(f"runs/gpt4o_level{level}/results.json")
    d4o = json.load(open(f4o))
    failed_problems = sorted([
        r["problem_id"] for r in d4o["per_problem"] if not r["correct"]
    ])

    run_prefix = f"gpt52pro_level{level}_failed"
    output_dir = Path(f"runs/{run_prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    solved = 0
    total = len(failed_problems)

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
        print(f"Resuming Level {level}: {len(already_done)} done, {solved} solved")

    print(f"\n{'#'*60}")
    print(f"  {MODEL} on Level {level} failures ({total} problems)")
    print(f"{'#'*60}")

    for i, pid in enumerate(failed_problems):
        if pid in already_done:
            continue

        print(f"\n[{i+1}/{total}] Level {level} P{pid}...", end=" ", flush=True)
        start = time.time()
        individual_name = f"{run_prefix}_p{pid}"

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "llm_kernel.cli", "run",
                    "--level", str(level),
                    "--problem", str(pid),
                    "--provider", "openai",
                    "--model", MODEL,
                    "--name", individual_name,
                    "--iterations", "3",  # fewer iterations due to slow model
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout (reasoning model is slow)
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
                        "problem_id": pid, "level": level, "correct": False,
                        "speedup": 0.0, "compile_error": True, "runtime_error": False,
                        "error_message": "No results"
                    })
                    print(f"no results in {elapsed:.0f}s")
            else:
                all_results.append({
                    "problem_id": pid, "level": level, "correct": False,
                    "speedup": 0.0, "compile_error": True, "runtime_error": False,
                    "error_message": "Results file not found"
                })
                print(f"no file in {elapsed:.0f}s")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            print(f"TIMEOUT ({elapsed:.0f}s)")
            all_results.append({
                "problem_id": pid, "level": level, "correct": False,
                "speedup": 0.0, "compile_error": False, "runtime_error": True,
                "error_message": "Timeout (600s)"
            })

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "problem_id": pid, "level": level, "correct": False,
                "speedup": 0.0, "compile_error": True, "runtime_error": False,
                "error_message": str(e)
            })

        # Save partial
        partial = {
            "run_name": run_prefix,
            "model": MODEL,
            "level": level,
            "iterations": 3,
            "total_problems": total,
            "solved_so_far": solved,
            "per_problem": all_results,
        }
        with open(results_file, 'w') as f:
            json.dump(partial, f, indent=2)

        print(f"Progress: {solved}/{len(all_results)} solved ({solved/max(len(all_results),1)*100:.1f}%)")

    # Save final
    final = {
        "run_name": run_prefix,
        "model": MODEL,
        "level": level,
        "total_problems": total,
        "solved_count": solved,
        "per_problem": all_results,
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(final, f, indent=2)

    print(f"\n>>> {MODEL} Level {level}: {solved}/{total} ({solved/max(total,1)*100:.1f}%)")

print(f"\n{'='*60}")
print(f"{MODEL} SUMMARY (Level 2/3)")
print(f"{'='*60}")
for level in [2, 3]:
    f = Path(f"runs/gpt52pro_level{level}_failed/results.json")
    if f.exists():
        d = json.load(open(f))
        solved = sum(1 for r in d["per_problem"] if r["correct"])
        total = len(d["per_problem"])
        print(f"  Level {level}: {solved}/{total}")
