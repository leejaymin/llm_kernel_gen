# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llm_kernel**: LLM 기반 Triton GPU 커널 자동 생성 및 KernelBench 평가 시스템. RTX PRO Blackwell (96GB) 타겟.

PyTorch 레퍼런스 코드 → LLM 생성 → Triton 커널 → 자동 평가 파이프라인.

## Commands

```bash
# 패키지 설치 (editable mode)
pip install -e ".[dev]"

# vLLM 서버 관리
llm-kernel server start --model qwen2.5-coder-32b
llm-kernel server status
llm-kernel server stop

# 커널 생성 및 평가 (KernelBench)
llm-kernel run --level 1 --model qwen2.5-coder-32b
llm-kernel run -l 2 -p 5 --samples 3            # Level 2, Problem 5, best-of-3
llm-kernel run --level 1 --strategy iterative   # 반복 개선 모드

# TritonBench 통합
llm-kernel tritonbench list                     # 채널 T (Triton) 문제 목록
llm-kernel tritonbench list --channel G         # 채널 G (General) 문제 목록
llm-kernel tritonbench list --stats             # 통계 포함
llm-kernel tritonbench run --name torch.matmul  # 특정 문제 실행

# 결과 분석
llm-kernel analyze <run_name> --detailed
llm-kernel analyze run1 --compare run2

# 모델 목록
llm-kernel models
llm-kernel models --fits  # 96GB에 맞는 모델만

# 테스트 (FastAPI 레거시)
pytest tests/ -v
pytest tests/test_tasks.py::TestCreateTask::test_create_task_minimal -v
```

## Architecture

### 핵심 파이프라인 (src/llm_kernel/)

```
KernelBench Problem
    ↓
[Generator] → LLM이 Triton 코드 생성 (GPU 힌트 포함)
    ↓
[Evaluator] → 정확성 테스트 + 성능 벤치마크
    ↓
[Reflector] → 에러 분석 및 분류
    ↓
[Optimizer] → 정적 분석 기반 최적화 제안
    ↓
[Orchestrator] → 상태 머신 (GENERATE → EVALUATE → REFLECT → OPTIMIZE)
    ↓
Generated Kernel + Metrics
```

### 모듈 구조

| 레이어 | 모듈 | 책임 |
|--------|------|------|
| CLI | `cli.py` | Typer 기반 명령어 인터페이스 |
| Bench | `bench/` | KernelBench 문제 로딩, 포맷팅, 실행, 결과 집계 |
| Client | `client/` | vLLM 클라이언트, Jinja2 프롬프트 빌더, 응답 파서 |
| Pipeline | `pipeline/` | 생성-평가-리플렉션-최적화 루프 오케스트레이션 |
| Knowledge | `knowledge/` | GPU 스펙, Triton 패턴, Blackwell 최적화 힌트 |
| Storage | `storage/` | 실행 결과 저장, 커널 캐시 |

### 의존성 흐름

```
cli.py
  ├── bench/problem_loader.py → HuggingFace datasets (KernelBench)
  ├── client/llm_client.py → OpenAI-compatible vLLM API
  └── pipeline/orchestrator.py
        ├── generator.py → llm_client + prompt_builder
        ├── evaluator.py → torch, importlib (동적 모듈 로딩)
        ├── reflector.py → knowledge/triton_patterns (에러 솔루션 DB)
        └── optimizer.py → knowledge/gpu_specs (GPU별 최적화)
```

### 레거시 API (app/)

FastAPI Task Management REST API. 메인 프로젝트와 분리됨.

- Routes → Services → Models 레이어 구조
- SQLite + SQLAlchemy
- `/api/v1/` 마운트, 응답 envelope: `{status, data, error}`

## Configuration

### 주요 설정 (config/default.yaml)

```yaml
gpu:
  name: "RTX PRO 6000 Blackwell"
  vram_gb: 96

llm:
  base_url: "http://localhost:8000/v1"
  default_model: "qwen2.5-coder-32b"

pipeline:
  strategy: "iterative"    # single_pass | iterative | best_of_n
  max_iterations: 5

kernelbench:
  n_correctness: 5         # 정확성 테스트 반복
  n_trial: 10              # 성능 벤치마크 trial
```

### 모델 레지스트리 (config/model/models.yaml)

15+ 모델 정의. 모델별 VRAM 요구량, dtype, max_model_len, 강점 문서화.

## Key Concepts

### 생성 전략
- **single_pass**: LLM 1회 호출
- **iterative**: 에러 피드백 기반 반복 개선 (최대 5회)
- **best_of_n**: N개 병렬 생성 후 최고 정확도 선택

### 평가 메트릭
- **Fast-0**: Correctness rate (정답률)
- **Fast-1**: Speedup ≥ 1.0x 비율
- **Fast-2**: Speedup ≥ 2.0x 비율

### 동적 코드 실행
생성된 Triton 코드는 tempfile + importlib로 격리된 환경에서 실행.

## Testing

테스트 파일은 `tests/` 디렉토리. FastAPI 레거시 API 대상.
- conftest.py: `get_db` 오버라이드, 테스트 DB 자동 생성/삭제
- pytest-asyncio: 비동기 테스트 지원

파이프라인 모듈 개별 테스트는 `notebooks/` 또는 직접 Python import 권장.
