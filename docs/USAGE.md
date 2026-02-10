# LLM Kernel - 사용 가이드

## 목차

- [환경 설정](#환경-설정)
- [커널 생성 및 평가 (KernelBench)](#커널-생성-및-평가-kernelbench)
- [OpenAI API 사용](#openai-api-사용)
- [서버 관리](#서버-관리)
- [결과 분석](#결과-분석)
- [TritonBench](#tritonbench)
- [모델 관리](#모델-관리)
- [CLI 옵션 전체 목록](#cli-옵션-전체-목록)
- [내부 변경사항](#내부-변경사항)

---

## 환경 설정

```bash
# conda 환경 활성화
conda activate llm_kernel

# 패키지 설치 (최초 1회 또는 코드 변경 후)
pip install -e ".[dev]"
```

---

## 커널 생성 및 평가 (KernelBench)

### 기본 실행 (로컬 vLLM)

```bash
# 1. vLLM 서버 시작 (별도 터미널)
vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000

# 2. Level 1 전체 실행
llm-kernel run --level 1 --name qwen32b_level1

# 3. 결과 확인
llm-kernel analyze qwen32b_level1 --detailed

# 결과 저장 위치: runs/qwen32b_level1/
```

### 특정 문제만 실행

```bash
# Level 1, Problem 5 (행렬-스칼라 곱)
llm-kernel run --level 1 --problem 5 --name test_p5

# Level 2, Problem 10
llm-kernel run --level 2 --problem 10 --name test_l2_p10
```

### 생성 전략 변경

```bash
# 반복 개선 모드 (기본값, 에러 피드백으로 재시도)
llm-kernel run --level 1 --strategy iterative --iterations 5

# 단일 생성 (1회만 시도)
llm-kernel run --level 1 --strategy single_pass

# Best-of-N (N개 생성 후 최고 선택)
llm-kernel run --level 1 --strategy best_of_n --samples 3
```

### 병렬 실행

```bash
# 워커 수 조절 (기본값: 4)
llm-kernel run --level 1 --workers 8 --name fast_run
```

---

## OpenAI API 사용

vLLM 로컬 서버 없이 OpenAI의 상용 GPT 모델로 커널을 생성합니다.

### 설정

```bash
# 방법 1: 환경변수 (권장)
export OPENAI_API_KEY="sk-..."

# 방법 2: 명령어에서 직접 전달
llm-kernel run --provider openai --openai-api-key "sk-..."
```

### 실행

```bash
# ============================================
# 💚 Tier 1: 초저가 (빠른 테스트 / 예산 절약)
# ============================================
# GPT-5 Nano — 가장 저렴하면서 코딩 벤치 78.9로 양호
llm-kernel run --level 1 --provider openai --model gpt-5-nano --name gpt5nano_level1

# ============================================
# 💛 Tier 2: 가성비 (추천 시작점)
# ============================================
# GPT-5 Mini — $0.25/$2.00인데 코딩 83.8, 가성비 최강 ⭐
llm-kernel run --level 1 --provider openai --model gpt-5-mini --name gpt5mini_level1

# GPT-4.1 Mini — 1M 컨텍스트, 긴 프롬프트 필요 시
llm-kernel run --level 1 --provider openai --model gpt-4.1-mini --name gpt41mini_level1

# ============================================
# 🟠 Tier 3: 중간 (고성능 코드 생성)
# ============================================
# o4-mini — 추론형 모델, 코딩 벤치 85.9 최고 가성비 추론
llm-kernel run --level 1 --provider openai --model o4-mini --name o4mini_level1

# GPT-5 — 범용 플래그십, 코딩 84.6
llm-kernel run --level 1 --provider openai --model gpt-5 --name gpt5_level1

# o3 — 추론형, 복잡한 Triton 커널에 적합
llm-kernel run --level 1 --provider openai --model o3 --name o3_level1

# GPT-4.1 — 1M 컨텍스트, RAG + 긴 레퍼런스 코드 포함 시
llm-kernel run --level 1 --provider openai --model gpt-4.1 --name gpt41_level1

# ============================================
# 🔴 Tier 4: 프리미엄 (최대 성능)
# ============================================
# GPT-5.2 — 최신 플래그십
llm-kernel run --level 1 --provider openai --model gpt-5.2 --name gpt52_level1

# GPT-5.2 Pro — 코딩 벤치 88.9, 최고 성능 (매우 비쌈)
llm-kernel run --level 1 --provider openai --model gpt-5.2-pro --name gpt52pro_level1

# ============================================
# 특정 문제만 테스트
# ============================================
llm-kernel run --level 1 --problem 5 --provider openai --model gpt-5-mini --name test_openai

```

### 지원 모델

| 모델 | 컨텍스트 | 특징 |
|------|----------|------|
| `gpt-4o` | 128K | 최신 최적화 모델 (권장) |
| `gpt-4-turbo` | 128K | 빠른 GPT-4 변형 |
| `gpt-4o-mini` | 128K | 경량, 비용 효율적 |
| `gpt-3.5-turbo` | 16K | 빠르고 경제적, 단순 작업 |

---

## 서버 관리

```bash
# 서버 시작
llm-kernel server start --model qwen2.5-coder-32b

# 서버 상태 확인
llm-kernel server status

# 서버 중지
llm-kernel server stop

# 사용 가능한 백엔드 확인 (vLLM, Ollama, 외부)
llm-kernel server check
```

### 서버 옵션

```bash
# GPU 메모리 사용률 조절
llm-kernel server start --model qwen2.5-coder-32b --gpu-util 0.8

# 포트 변경
llm-kernel server start --model qwen2.5-coder-32b --port 8080

# Tensor Parallel (멀티 GPU)
llm-kernel server start --model qwen2.5-coder-32b --tp 2
```

### 외부 서버 사용

```bash
# 다른 서버에서 실행 중인 vLLM 사용
llm-kernel run --level 1 --url http://other-server:8000/v1 --name remote_run
```

---

## 결과 분석

```bash
# 기본 요약
llm-kernel analyze qwen32b_level1

# 상세 결과 (문제별)
llm-kernel analyze qwen32b_level1 --detailed

# 두 실행 비교
llm-kernel analyze qwen32b_level1 --compare gpt4o_level1

# CSV 내보내기
llm-kernel analyze qwen32b_level1 --csv
```

### 결과 파일 구조

```
runs/
  qwen32b_level1/
    problem_1.py        # 생성된 Triton 커널
    problem_2.py
    ...
    results.json        # 전체 결과 (JSON)
    results.csv         # 전체 결과 (CSV)
```

---

## TritonBench

[thunlp/TritonBench](https://github.com/thunlp/TritonBench) 벤치마크를 조회하고 실행합니다.

### 문제 목록 조회

```bash
# 채널 T (Triton 전문) - 166개 문제
llm-kernel tritonbench list

# 통계 포함
llm-kernel tritonbench list --stats

# 채널 G (General) - 184개 문제
llm-kernel tritonbench list --channel G

# 채널 G 통계 포함
llm-kernel tritonbench list --channel G --stats
```

### 특정 문제 조회

```bash
# 문제 상세 정보 + 레퍼런스 코드
llm-kernel tritonbench run --name torch.matmul
llm-kernel tritonbench run --name lightning_attention --channel G
```

### 채널 설명

| 채널 | 설명 | 문제 수 |
|------|------|---------|
| T | Triton 전문 벤치마크 | 166 |
| G | General GPU 벤치마크 | 184 |

---

## 모델 관리

```bash
# 전체 모델 목록 (VRAM 정보 포함)
llm-kernel models

# 96GB GPU에 맞는 모델만
llm-kernel models --fits
```

---

## CLI 옵션 전체 목록

### `llm-kernel run`

| 옵션 | 단축 | 기본값 | 설명 |
|------|------|--------|------|
| `--level` | `-l` | 1 | KernelBench 레벨 (1-4) |
| `--problem` | `-p` | 전체 | 특정 문제 ID |
| `--model` | `-m` | qwen2.5-coder-32b | 모델명 |
| `--provider` | | vllm | LLM 제공자 (vllm / openai) |
| `--strategy` | `-s` | iterative | 생성 전략 |
| `--iterations` | `-i` | 5 | 최대 반복 횟수 |
| `--samples` | `-n` | 1 | Best-of-N 샘플 수 |
| `--name` | | 자동생성 | 실행 이름 |
| `--workers` | `-w` | 4 | 병렬 워커 수 |
| `--url` | | localhost:8000/v1 | vLLM 서버 URL |
| `--openai-api-key` | | env var | OpenAI API 키 |
| `--output` | `-o` | runs | 결과 저장 디렉토리 |

### `llm-kernel tritonbench`

| 옵션 | 단축 | 기본값 | 설명 |
|------|------|--------|------|
| action (인자) | | 필수 | list 또는 run |
| `--name` | `-n` | | 문제 이름 (run 시 필수) |
| `--channel` | `-c` | T | 채널 (T / G) |
| `--stats` | | false | 통계 표시 |

### `llm-kernel analyze`

| 옵션 | 단축 | 기본값 | 설명 |
|------|------|--------|------|
| run_name (인자) | | 필수 | 분석할 실행 이름 |
| `--compare` | `-c` | | 비교 대상 실행 |
| `--detailed` | `-d` | false | 문제별 상세 결과 |
| `--csv` | | false | CSV 내보내기 |

---

## 내부 변경사항

기존 명령어를 그대로 사용해도 다음 개선이 자동 적용됩니다:

### 시스템 프롬프트 추가

LLM 호출 시 system role에 Triton 전문가 역할 및 출력 포맷 규칙이 포함됩니다:
- `class Model(nn.Module)` 필수 사용
- `get_inputs()`에 CUDA 텐서 반환 필수
- 코드블록 외 설명 텍스트 금지

### 프롬프트 템플릿 개선

- few-shot 예제 (완전한 Triton 커널 예시) 포함
- 출력 포맷 규칙 강화

### 응답 후처리 강화

| 후처리 | 설명 |
|--------|------|
| 클래스명 변환 | TritonModel, MyModel 등 → Model 자동 변환 |
| nn.Module 상속 | `class Model:` → `class Model(nn.Module):` |
| 설명 텍스트 제거 | `### Explanation:` 등 13가지 패턴 자동 제거 |
| main 블록 제거 | `if __name__ == '__main__'` 자동 제거 |
| CUDA 검증 | get_inputs()에 `.cuda()` 미사용 시 경고 |
