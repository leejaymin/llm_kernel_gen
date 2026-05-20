#!/usr/bin/env python3
"""
Mindlogic Gateway를 통한 다중 모델 KernelBench 실험 스크립트.

비용 효율적인 최신 모델들 위주로 Level 1 실험을 수행합니다.
환경변수 MINDLOGIC_API_KEY가 설정되어 있어야 합니다.

사용법:
    source examples/setup_mindlogic.sh
    python run_mindlogic_experiments.py [--models MODEL1,MODEL2,...] [--resume]
"""

import json
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path


# ============================================================
# 비용 효율적 모델 목록 (저비용 → 중비용 순)
# ============================================================
ALL_MODELS = {
    # Tier 1: 저비용 소형 모델
    "gpt-5.4-nano": {"tier": "저비용", "iterations": 3},
    "gemini-3.1-flash-lite-preview": {"tier": "저비용", "iterations": 3},
    "claude-haiku-4-5-20251001": {"tier": "저비용", "iterations": 3},
    "google/gemma-4-31B-it": {"tier": "저비용", "iterations": 3},
    "solar-pro3": {"tier": "저비용", "iterations": 3},
    # Tier 2: 중비용 고성능 모델
    "gpt-5.4-mini": {"tier": "중비용", "iterations": 3},
    "gemini-2.5-flash": {"tier": "중비용", "iterations": 3},
    "grok-3-mini": {"tier": "중비용", "iterations": 3},
    "claude-sonnet-4-6": {"tier": "중비용", "iterations": 3},
}


def make_run_name(model: str) -> str:
    """모델명을 안전한 디렉토리명으로 변환."""
    return "mindlogic_" + model.replace("/", "_").replace("-", "_").replace(".", "") + "_L1"


def run_single_problem(model: str, level: int, pid: int, iterations: int, run_dir: Path) -> dict:
    """단일 문제를 CLI subprocess로 실행하고 결과를 반환."""
    sub_name = f"{run_dir.name}_p{pid}"

    cmd = [
        sys.executable, "-m", "llm_kernel.cli", "run",
        "--level", str(level),
        "--problem", str(pid),
        "--provider", "mindlogic",
        "--model", model,
        "--name", sub_name,
        "--iterations", str(iterations),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min per problem
            env={**os.environ, "MINDLOGIC_API_KEY": os.environ.get("MINDLOGIC_API_KEY", "")},
            cwd=str(Path(__file__).parent),
        )

        # Read individual result
        individual_path = Path("runs") / sub_name / "results.json"
        if individual_path.exists():
            with open(individual_path) as f:
                data = json.load(f)
            if data.get("per_problem"):
                return data["per_problem"][0]

    except subprocess.TimeoutExpired:
        return {
            "problem_id": pid, "level": level, "correct": False,
            "speedup": 0.0, "compile_error": False, "runtime_error": True,
            "error_message": "Timeout (300s)"
        }
    except Exception as e:
        return {
            "problem_id": pid, "level": level, "correct": False,
            "speedup": 0.0, "compile_error": True, "runtime_error": False,
            "error_message": str(e)
        }

    return {
        "problem_id": pid, "level": level, "correct": False,
        "speedup": 0.0, "compile_error": True, "runtime_error": False,
        "error_message": "No results file"
    }


def run_model_experiment(model: str, config: dict, level: int = 1) -> dict:
    """한 모델에 대해 전체 레벨 실험 실행."""
    run_name = make_run_name(model)
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "results.json"
    partial_file = run_dir / "results_partial.json"

    # Resume: 이미 완료된 문제 건너뛰기
    all_results = []
    already_done = set()
    if partial_file.exists():
        with open(partial_file) as f:
            existing = json.load(f)
        for r in existing.get("per_problem", []):
            already_done.add(r["problem_id"])
            all_results.append(r)
        print(f"  [Resume] {len(already_done)}개 문제 완료됨, {sum(1 for r in all_results if r['correct'])}개 해결")

    # Level 1 = 100 problems (pid 1-100)
    total_problems = 100
    solved = sum(1 for r in all_results if r["correct"])
    iterations = config["iterations"]

    for pid in range(1, total_problems + 1):
        if pid in already_done:
            continue

        t0 = time.time()
        r = run_single_problem(model, level, pid, iterations, run_dir)
        elapsed = time.time() - t0

        all_results.append(r)

        if r["correct"]:
            solved += 1
            speedup = r.get("speedup", 0)
            print(f"  P{pid:>3}: ✓ {speedup:.2f}x ({elapsed:.0f}s) [{solved}/{len(all_results)}]")
        else:
            err_type = "Compile" if r.get("compile_error") else ("Timeout" if "Timeout" in r.get("error_message", "") else "Runtime")
            print(f"  P{pid:>3}: ✗ {err_type} ({elapsed:.0f}s) [{solved}/{len(all_results)}]")

        # 매 문제마다 중간 결과 저장 (resume 지원)
        partial_data = {
            "run_name": run_name,
            "model": model,
            "level": level,
            "total_problems": total_problems,
            "solved_count": solved,
            "per_problem": sorted(all_results, key=lambda x: x["problem_id"]),
        }
        with open(partial_file, "w") as f:
            json.dump(partial_data, f, indent=2)

    # 최종 결과 저장
    final_data = {
        "run_name": run_name,
        "model": model,
        "level": level,
        "total_problems": total_problems,
        "solved_count": solved,
        "per_problem": sorted(all_results, key=lambda x: x["problem_id"]),
    }
    with open(results_file, "w") as f:
        json.dump(final_data, f, indent=2)

    return {"model": model, "tier": config["tier"], "solved": solved, "total": total_problems}


def main():
    parser = argparse.ArgumentParser(description="Mindlogic Gateway 다중 모델 KernelBench 실험")
    parser.add_argument("--models", type=str, default=None,
                       help="실행할 모델 (쉼표 구분). 미지정 시 전체 실행")
    parser.add_argument("--level", type=int, default=1, help="KernelBench 레벨 (기본: 1)")
    parser.add_argument("--list", action="store_true", help="사용 가능한 모델 목록 표시")
    args = parser.parse_args()

    if args.list:
        print("\n사용 가능한 모델:")
        for m, c in ALL_MODELS.items():
            print(f"  [{c['tier']}] {m}")
        return

    # API 키 확인
    if not os.getenv("MINDLOGIC_API_KEY"):
        print("ERROR: MINDLOGIC_API_KEY 환경변수가 설정되지 않았습니다.")
        print("  source examples/setup_mindlogic.sh")
        sys.exit(1)

    # 모델 선정
    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        models = {m: ALL_MODELS[m] for m in selected if m in ALL_MODELS}
        # 목록에 없는 모델도 지원 (기본 설정으로)
        for m in selected:
            if m not in models:
                models[m] = {"tier": "커스텀", "iterations": 3}
    else:
        models = ALL_MODELS

    print("=" * 70)
    print(f"Mindlogic Gateway - KernelBench Level {args.level} 다중 모델 실험")
    print(f"시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"모델: {len(models)}개")
    print("=" * 70)

    for i, (m, c) in enumerate(models.items(), 1):
        print(f"  {i}. [{c['tier']}] {m}")
    print()

    # 실험 실행
    summary = []
    for i, (model, config) in enumerate(models.items(), 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(models)}] {model} ({config['tier']})")
        print(f"{'='*70}")

        t0 = time.time()
        result = run_model_experiment(model, config, level=args.level)
        elapsed = time.time() - t0

        result["elapsed_min"] = f"{elapsed/60:.1f}"
        summary.append(result)

        print(f"\n  완료: {result['solved']}/{result['total']} ({elapsed/60:.1f}분)")

    # 최종 요약
    print(f"\n{'='*70}")
    print("실험 결과 요약")
    print(f"{'='*70}")
    print(f"\n{'Model':<40} {'Tier':<8} {'Solved':>10} {'Time':>8}")
    print("-" * 70)
    for r in sorted(summary, key=lambda x: x["solved"], reverse=True):
        print(f"{r['model']:<40} {r['tier']:<8} {r['solved']:>3}/{r['total']:<5} {r['elapsed_min']:>7}m")

    # 요약 JSON 저장
    summary_path = Path("runs") / "mindlogic_experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "level": args.level,
            "experiments": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
