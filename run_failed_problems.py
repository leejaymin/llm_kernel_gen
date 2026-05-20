#!/usr/bin/env python3
"""Run gpt-5.2-pro on problems that gpt-4o failed to solve."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Load gpt-4o results to find failed problems
results_path = Path("runs/gpt4o_level1_v2/results.json")
with open(results_path) as f:
    data = json.load(f)

failed_problems = sorted([
    r["problem_id"] for r in data["per_problem"] if not r["correct"]
])

print(f"Total failed problems from gpt-4o: {len(failed_problems)}")
print(f"Problems: {failed_problems}")

# Output directory for gpt-5.2-pro results
output_dir = Path("runs/gpt-5.2-pro_failed")
output_dir.mkdir(parents=True, exist_ok=True)

# Track results
all_results = []
solved = 0
total = len(failed_problems)

# Check which problems are already solved (resume support)
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
    print(f"Resuming: {len(already_done)} problems already done, {solved} solved")

# Run each failed problem
for i, pid in enumerate(failed_problems):
    if pid in already_done:
        continue

    print(f"\n{'='*60}")
    print(f"[{i+1}/{total}] Running Problem {pid} with gpt-5.2-pro...")
    print(f"{'='*60}")

    start = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "llm_kernel.cli", "run",
                "--level", "1",
                "--problem", str(pid),
                "--provider", "openai",
                "--model", "gpt-5.2-pro",
                "--name", f"gpt-5.2-pro_p{pid}",
                "--iterations", "3",
            ],
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min timeout per problem
            env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
            cwd=str(Path(__file__).parent),
        )

        elapsed = time.time() - start
        print(f"Completed in {elapsed:.1f}s")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

        if result.stderr:
            # Filter out ResourceTracker noise
            stderr_lines = [l for l in result.stderr.split('\n')
                          if 'ResourceTracker' not in l and '_recursion_count' not in l and l.strip()]
            if stderr_lines:
                print("STDERR:", '\n'.join(stderr_lines[-5:]))

        # Try to read the individual result
        individual_result_path = Path(f"runs/gpt-5.2-pro_p{pid}/results.json")
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
                    print(f"  >>> Failed: compile_error={r.get('compile_error')}, runtime_error={r.get('runtime_error')}")
            else:
                all_results.append({
                    "problem_id": pid, "level": 1, "correct": False,
                    "speedup": 0.0, "compile_error": True, "runtime_error": False,
                    "error_message": "No results"
                })
        else:
            all_results.append({
                "problem_id": pid, "level": 1, "correct": False,
                "speedup": 0.0, "compile_error": True, "runtime_error": False,
                "error_message": "Results file not found"
            })

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  TIMEOUT after {elapsed:.1f}s")
        all_results.append({
            "problem_id": pid, "level": 1, "correct": False,
            "speedup": 0.0, "compile_error": False, "runtime_error": True,
            "error_message": "Timeout (1200s)"
        })

    except Exception as e:
        print(f"  ERROR: {e}")
        all_results.append({
            "problem_id": pid, "level": 1, "correct": False,
            "speedup": 0.0, "compile_error": True, "runtime_error": False,
            "error_message": str(e)
        })

    # Save partial results after each problem
    partial = {
        "run_name": "gpt-5.2-pro_failed",
        "level": 1,
        "total_problems": total,
        "solved_so_far": solved,
        "progress": f"{len([r for r in all_results if r['problem_id'] not in already_done or r['problem_id'] in already_done])}/{total}",
        "per_problem": all_results,
    }
    with open(results_file, 'w') as f:
        json.dump(partial, f, indent=2)

    print(f"\nProgress: {solved}/{len(all_results)} solved ({solved/max(len(all_results),1)*100:.1f}%)")

# Final summary
print(f"\n{'='*60}")
print(f"FINAL RESULTS: gpt-5.2-pro on gpt-4o failed problems")
print(f"{'='*60}")
print(f"Total failed problems from gpt-4o: {total}")
print(f"Solved by gpt-5.2-pro: {solved} ({solved/total*100:.1f}%)")
print(f"Still unsolved: {total - solved}")

# Combined stats (gpt-4o solved + gpt-5.2-pro newly solved)
gpt4o_solved = 19
combined = gpt4o_solved + solved
print(f"\nCombined (gpt-4o + gpt-5.2-pro): {combined}/100 ({combined}%)")
print(f"Improvement: {gpt4o_solved}% -> {combined}%")

# Save final results
final = {
    "run_name": "gpt-5.2-pro_failed",
    "level": 1,
    "total_problems": total,
    "solved_by_5_2_pro": solved,
    "gpt4o_baseline": gpt4o_solved,
    "combined_score": combined,
    "per_problem": all_results,
}
with open(output_dir / "results_final.json", 'w') as f:
    json.dump(final, f, indent=2)
