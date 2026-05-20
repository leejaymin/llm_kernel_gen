#!/usr/bin/env python3
"""
Qwen 2.5 Coder 32B 재실험 스크립트.
vLLM 로컬 서버를 통해 Level 1 실험 수행.

사용법:
    # vLLM 서버가 실행 중이어야 함
    python run_qwen_experiment.py
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
RUN_NAME = "qwen25_coder_32b_L1"
LEVEL = 1
ITERATIONS = 3
TIMEOUT_PER_PROBLEM = 300  # 5 min


def run_single_problem(pid: int) -> dict:
    sub_name = f"{RUN_NAME}_p{pid}"
    cmd = [
        sys.executable, "-m", "llm_kernel.cli", "run",
        "--level", str(LEVEL),
        "--problem", str(pid),
        "--provider", "vllm",
        "--model", MODEL,
        "--url", "http://localhost:8000/v1",
        "--name", sub_name,
        "--iterations", str(ITERATIONS),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_PROBLEM,
            cwd=str(Path(__file__).parent),
        )

        individual_path = Path("runs") / sub_name / "results.json"
        if individual_path.exists():
            with open(individual_path) as f:
                data = json.load(f)
            if data.get("per_problem"):
                return data["per_problem"][0]

    except subprocess.TimeoutExpired:
        return {
            "problem_id": pid, "level": LEVEL, "correct": False,
            "speedup": 0.0, "compile_error": False, "runtime_error": True,
            "error_message": "Timeout (300s)"
        }
    except Exception as e:
        return {
            "problem_id": pid, "level": LEVEL, "correct": False,
            "speedup": 0.0, "compile_error": True, "runtime_error": False,
            "error_message": str(e)
        }

    return {
        "problem_id": pid, "level": LEVEL, "correct": False,
        "speedup": 0.0, "compile_error": True, "runtime_error": False,
        "error_message": "No results file"
    }


def main():
    run_dir = Path("runs") / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    partial_file = run_dir / "results_partial.json"
    results_file = run_dir / "results.json"

    # Resume support
    all_results = []
    already_done = set()
    if partial_file.exists():
        with open(partial_file) as f:
            existing = json.load(f)
        for r in existing.get("per_problem", []):
            already_done.add(r["problem_id"])
            all_results.append(r)
        solved = sum(1 for r in all_results if r["correct"])
        print(f"[Resume] {len(already_done)}개 완료, {solved}개 해결")

    total_problems = 100
    solved = sum(1 for r in all_results if r["correct"])

    print(f"{'='*70}")
    print(f"Qwen 2.5 Coder 32B - KernelBench Level {LEVEL}")
    print(f"Model: {MODEL}")
    print(f"Iterations: {ITERATIONS}")
    print(f"시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for pid in range(1, total_problems + 1):
        if pid in already_done:
            continue

        t0 = time.time()
        r = run_single_problem(pid)
        elapsed = time.time() - t0

        all_results.append(r)

        if r["correct"]:
            solved += 1
            speedup = r.get("speedup", 0)
            print(f"  P{pid:>3}: ✓ {speedup:.2f}x ({elapsed:.0f}s) [{solved}/{len(all_results)}]")
        else:
            err_type = "Compile" if r.get("compile_error") else ("Timeout" if "Timeout" in r.get("error_message", "") else "Runtime")
            print(f"  P{pid:>3}: ✗ {err_type} ({elapsed:.0f}s) [{solved}/{len(all_results)}]")

        # Save partial results
        partial_data = {
            "run_name": RUN_NAME,
            "model": MODEL,
            "level": LEVEL,
            "total_problems": total_problems,
            "solved_count": solved,
            "per_problem": sorted(all_results, key=lambda x: x["problem_id"]),
        }
        with open(partial_file, "w") as f:
            json.dump(partial_data, f, indent=2)

    # Final save
    final_data = {
        "run_name": RUN_NAME,
        "model": MODEL,
        "level": LEVEL,
        "total_problems": total_problems,
        "solved_count": solved,
        "per_problem": sorted(all_results, key=lambda x: x["problem_id"]),
    }
    with open(results_file, "w") as f:
        json.dump(final_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"완료: {solved}/{total_problems} ({solved}%)")
    print(f"저장: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
