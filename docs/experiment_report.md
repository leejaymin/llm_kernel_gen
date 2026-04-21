# LLM-based Triton GPU Kernel Generation: Experiment Report

> **Date:** 2026-03-01
> **System:** NVIDIA RTX PRO 6000 Blackwell (96GB), sm_120
> **Benchmark:** KernelBench (Level 1: 100, Level 2: 100, Level 3: 50 problems)
> **Framework:** llm_kernel — PyTorch reference code to Triton kernel auto-generation with iterative refinement
> **Evaluation Metric:** Fast-0 (correctness rate, tolerance 1e-2)

---

## Table of Contents

1. [Experimental Setup](#1-experimental-setup)
2. [Model Comparison Results](#2-model-comparison-results)
3. [Multi-Model Cascading Strategy](#3-multi-model-cascading-strategy)
4. [Iteration Ablation Study](#4-iteration-ablation-study)
5. [Error Analysis](#5-error-analysis)
6. [Speedup Analysis](#6-speedup-analysis)
7. [Per-Problem Detailed Results](#7-per-problem-detailed-results)
8. [Key Findings](#8-key-findings)
9. [Data Source Reference](#9-data-source-reference)

---

## 1. Experimental Setup

### 1.1 Pipeline Architecture

```
KernelBench Problem (PyTorch reference code)
    |
    v
[PromptBuilder] -- GPU specs + Triton patterns + system prompt
    |
    v
[LLM Generation] -- OpenAI API (gpt-4o-mini / gpt-4o / gpt-5.2 / gpt-5.2-pro)
    |
    v
[ResponseParser] -- Code extraction, class rename, prose removal
    |
    v
[Evaluator] -- Subprocess isolation (CUDA error prevention)
    |   - Correctness: 5 random inputs, tolerance 1e-2
    |   - Performance: 10 benchmark trials
    |
    v
[Reflector] -- Error classification + feedback generation
    |
    v
[Iterative Refinement] -- Re-generate with error context (up to N iterations)
```

### 1.2 Models Tested

| Model | Type | API | Iterations | Timeout/problem |
|-------|------|-----|-----------|-----------------|
| gpt-4o-mini | Chat completion | Chat Completions | 5 | 300s |
| gpt-4o | Chat completion | Chat Completions | 5 | 300s |
| gpt-5.2 | Chat completion | Chat Completions | 5 | 300s |
| gpt-5.2-pro | Reasoning model | Responses API | 3 | 600s (L2/L3), 1200s (L1) |

### 1.3 Experimental Design

- **Experiment 1 (Model Comparison):** Each model on all 3 levels, gpt-5.2 and gpt-5.2-pro run on gpt-4o failure set only (cascading strategy)
- **Experiment 2 (Iteration Ablation):** gpt-4o on Level 1 with iterations = {1, 3, 5, 10}
- **Experiment 3 (Error Analysis):** Error categorization across all runs

### 1.4 Evaluation Criteria

A problem is considered **solved** (correct) when:
1. Generated Triton code compiles without error
2. Output matches PyTorch reference within tolerance (atol=1e-2) across 5 random inputs
3. No runtime errors (CUDA OOM, illegal memory access, etc.)

**Speedup** is measured as: `reference_time / generated_kernel_time` over 10 trials.

---

## 2. Model Comparison Results

### 2.1 Individual Model Performance (Fast-0: Correctness Rate)

| Model | Level 1 (100) | Level 2 (100) | Level 3 (50) | Total (250) |
|-------|:------------:|:------------:|:-----------:|:-----------:|
| gpt-4o-mini | 15 (15.0%) | 1 (1.0%) | 0 (0.0%) | 16 (6.4%) |
| gpt-4o | 19 (19.0%) | 6 (6.0%) | 0 (0.0%) | 25 (10.0%) |
| gpt-5.2* | 23/81 (28.4%) | 9/94 (9.6%) | 4/50 (8.0%) | 36/225 (16.0%) |
| gpt-5.2-pro* | 13/81 (16.0%) | 0/94 (0.0%) | 0/50 (0.0%) | 13/225 (5.8%) |

> \* gpt-5.2 and gpt-5.2-pro were run only on problems that gpt-4o failed to solve (cascading strategy). The denominators reflect the number of attempted problems.

**Data source:** See [Section 9.1](#91-primary-results-files) for exact file paths.

### 2.2 Observations

1. **gpt-5.2 achieves the highest per-attempt success rate** across all levels, with 28.4% on Level 1 failures and the only model to solve any Level 3 problems.
2. **gpt-5.2-pro (reasoning model) underperforms significantly** — 16% on Level 1 (vs. 28.4% for gpt-5.2), 0% on Level 2/3. On Level 2, 81.9% of problems timed out (600s) and 18.1% failed due to API rate limits. On Level 3, 100% failed due to API rate limits.
3. **Model capability scaling is non-linear:** gpt-4o-mini to gpt-4o shows ~1.6x improvement on Level 1, while gpt-4o to gpt-5.2 shows ~1.5x improvement (on the harder failure subset).
4. **Level 3 represents a significant difficulty cliff:** Both gpt-4o and gpt-4o-mini score 0/50. Only gpt-5.2 manages 4 solutions.

---

## 3. Multi-Model Cascading Strategy

### 3.1 Cascading Protocol

```
Phase 1: Run gpt-4o on ALL problems
Phase 2: Run gpt-5.2 on gpt-4o failures only
Phase 3: Run gpt-5.2-pro on remaining failures
Result:  UNION of all solved sets
```

### 3.2 Combined Results

| Configuration | L1 (/100) | L2 (/100) | L3 (/50) | Total (/250) |
|---------------|:---------:|:---------:|:--------:|:------------:|
| gpt-4o-mini (baseline) | 15 (15%) | 1 (1%) | 0 (0%) | 16 (6.4%) |
| gpt-4o | 19 (19%) | 6 (6%) | 0 (0%) | 25 (10.0%) |
| gpt-4o + gpt-5.2 | **42 (42%)** | **15 (15%)** | **4 (8%)** | **61 (24.4%)** |
| gpt-4o + gpt-5.2 + gpt-5.2-pro | **43 (43%)** | **15 (15%)** | **4 (8%)** | **62 (24.8%)** |

### 3.3 Marginal Contribution Analysis

| Model | L1 Unique | L2 Unique | L3 Unique | Total Unique |
|-------|:---------:|:---------:|:---------:|:------------:|
| gpt-4o | 19 | 6 | 0 | 25 |
| gpt-5.2 (additional) | +23 | +9 | +4 | +36 |
| gpt-5.2-pro (additional) | +1 | +0 | +0 | +1 |

- **gpt-5.2 adds 36 problems** beyond gpt-4o, representing a 2.44x improvement (25 → 61).
- **gpt-5.2-pro adds only 1 problem** (Level 1, Problem 88 with 7.60x speedup), offering minimal value beyond the gpt-4o + gpt-5.2 combination.

### 3.4 Solved Problem ID Sets

**Level 1 — gpt-4o solved (19 problems):**
`{2, 5, 6, 7, 8, 9, 12, 13, 15, 19, 20, 21, 22, 25, 27, 29, 30, 31, 32}`

> Data source: `runs/gpt4o_level1_v2/results.json` — filter `per_problem[].correct == true`

**Level 1 — gpt-5.2 additionally solved (23 problems):**
`{1, 3, 10, 11, 16, 17, 18, 23, 26, 28, 35, 36, 37, 40, 41, 42, 43, 44, 89, 95, 98, 99, 100}`

> Data source: `runs/gpt-5.2_failed/results_final.json` — filter `per_problem[].correct == true`

**Level 1 — gpt-5.2-pro additionally solved (1 problem):**
`{88}`

> Data source: `runs/gpt-5.2-pro_failed/results_final.json` — filter `per_problem[].correct == true`, exclude those in gpt-4o or gpt-5.2 solved sets

**Level 1 — Combined union (43 problems):**
`{1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 40, 41, 42, 43, 44, 88, 89, 95, 98, 99, 100}`

**Level 2 — gpt-4o solved (6 problems):**
`{9, 18, 38, 49, 70, 88}`

> Data source: `runs/gpt4o_level2/results.json`

**Level 2 — gpt-5.2 additionally solved (9 problems):**
`{23, 40, 45, 51, 64, 66, 80, 83, 99}`

> Data source: `runs/gpt52_level2_failed/results.json`

**Level 2 — Combined union (15 problems):**
`{9, 18, 23, 38, 40, 45, 49, 51, 64, 66, 70, 80, 83, 88, 99}`

**Level 3 — gpt-5.2 solved (4 problems, only model with solutions):**
`{16, 38, 43, 46}`

> Data source: `runs/gpt52_level3_failed/results.json`

---

## 4. Iteration Ablation Study

### 4.1 Setup

All runs use **gpt-4o on Level 1 (100 problems)** with varying iteration counts. The `iter=5` condition uses the baseline run (`gpt4o_level1_v2`).

### 4.2 Results

| Iterations | Solved (/100) | Rate | Compile Error% | Value Mismatch% |
|:----------:|:------------:|:----:|:--------------:|:---------------:|
| 1 | 19 | 19% | 59.3% | 24.7% |
| 3 | 21 | 21% | 51.9% | 39.2% |
| 5 | 19 | 19% | 45.7% | 37.0% |
| 10 | 22 | 22% | 32.1% | 53.8% |
| **Union** | **28** | **28%** | — | — |

### 4.3 Per-Iteration Solved Sets

| Iterations | Solved Problem IDs |
|:----------:|-------------------|
| 1 | {1, 2, 5, 6, 9, 12, 13, 14, 15, 19, 20, 21, 25, 27, 29, 30, 31, 32, 41} |
| 3 | {1, 2, 5, 6, 7, 8, 9, 12, 19, 20, 21, 25, 26, 27, 29, 30, 31, 32, 33, 41, 98} |
| 5 | {2, 5, 6, 7, 8, 9, 12, 13, 15, 19, 20, 21, 22, 25, 27, 29, 30, 31, 32} |
| 10 | {3, 5, 6, 7, 8, 9, 12, 13, 15, 19, 20, 21, 22, 25, 27, 29, 30, 31, 32, 35, 41, 95} |
| **Union** | {1, 2, 3, 5, 6, 7, 8, 9, 12, 13, 14, 15, 19, 20, 21, 22, 25, 26, 27, 29, 30, 31, 32, 33, 35, 41, 95, 98} |

> Data sources:
> - iter=1: `runs/ablation_iter1/results.json`
> - iter=3: `runs/ablation_iter3/results.json`
> - iter=5: `runs/gpt4o_level1_v2/results.json`
> - iter=10: `runs/ablation_iter10/results.json`

### 4.4 Analysis

1. **Marginal returns are diminishing:** Increasing from 1 to 10 iterations yields only +3 problems (19 → 22), a 15.8% improvement.
2. **Union significantly exceeds individual best:** The union of all 4 configurations (28) exceeds the best single run (22) by 27.3%. This demonstrates that **stochastic diversity across runs is more valuable than iteration depth within a single run**.
3. **Error profile shifts with iterations:**
   - Compile errors decrease: 59.3% (iter=1) → 32.1% (iter=10)
   - Value mismatch increases: 24.7% (iter=1) → 53.8% (iter=10)
   - Interpretation: Iterative refinement effectively fixes syntax/compilation issues but exposes deeper logical correctness problems that the model cannot resolve through error feedback alone.
4. **Problems uniquely solved by specific iteration counts:**
   - Only iter=1: {14}
   - Only iter=3: {26, 33, 98}
   - Only iter=5: (none unique — all are subsets of other runs)
   - Only iter=10: {3, 35, 95}
   - This suggests running multiple configurations with lower iterations may be more cost-effective than a single high-iteration run.

---

## 5. Error Analysis

### 5.1 Error Taxonomy

| Error Category | Description | Identification |
|---------------|-------------|----------------|
| **Compile Error** | Triton syntax error, Python compilation failure, invalid Triton API usage | `compile_error == true` in results |
| **Value Mismatch** | Code compiles and runs but produces incorrect output | `error_message` contains "mismatch", "Values", or "incorrect" |
| **CUDA Error** | Illegal memory access, device-side assert | `error_message` contains "CUDA" or "illegal" |
| **Timeout** | Exceeded time limit (300s-1200s depending on model) | `error_message` contains "Timeout" |
| **OOM** | GPU out-of-memory | `error_message` contains "OOM" or "out of memory" |
| **API Rate Limit** | OpenAI HTTP 429 error | `error_message` contains "429" or "rate" |

### 5.2 Error Distribution by Model and Level

#### gpt-4o-mini

| Level | Failed | Compile | Val Mismatch | CUDA | Timeout | OOM | Other |
|:-----:|:------:|:-------:|:------------:|:----:|:-------:|:---:|:-----:|
| 1 | 85 | 61 (71.8%) | 17 (20.0%) | 2 (2.4%) | 2 (2.4%) | 1 (1.2%) | 2 (2.4%) |
| 2 | 99 | 70 (70.7%) | 22 (22.2%) | 4 (4.0%) | 1 (1.0%) | 0 | 2 (2.0%) |
| 3 | 50 | 41 (82.0%) | 3 (6.0%) | 3 (6.0%) | 0 | 0 | 3 (6.0%) |

> Data sources: `runs/gpt4o_mini_level{1,2,3}/results.json` — inspect `per_problem[].error_message` for failed entries

#### gpt-4o

| Level | Failed | Compile | Val Mismatch | CUDA | Timeout | OOM | Other |
|:-----:|:------:|:-------:|:------------:|:----:|:-------:|:---:|:-----:|
| 1 | 81 | 37 (45.7%) | 30 (37.0%) | 9 (11.1%) | 0 | 0 | 5 (6.2%) |
| 2 | 94 | 48 (51.1%) | 39 (41.5%) | 4 (4.3%) | 0 | 0 | 3 (3.2%) |
| 3 | 50 | 19 (38.0%) | 28 (56.0%) | 2 (4.0%) | 0 | 0 | 1 (2.0%) |

> Data sources: `runs/gpt4o_level1_v2/results.json`, `runs/gpt4o_level{2,3}/results.json`

#### gpt-5.2

| Level | Failed | Compile | Val Mismatch | CUDA | Timeout | OOM | Other |
|:-----:|:------:|:-------:|:------------:|:----:|:-------:|:---:|:-----:|
| 1 | 58 | 4 (6.9%) | 34 (58.6%) | 1 (1.7%) | 12 (20.7%) | 1 (1.7%) | 6 (10.3%) |
| 2 | 85 | 22 (25.9%) | 53 (62.4%) | 0 | 7 (8.2%) | 0 | 3 (3.5%) |
| 3 | 46 | 8 (17.4%) | 33 (71.7%) | 0 | 2 (4.3%) | 0 | 3 (6.5%) |

> Data sources: `runs/gpt-5.2_failed/results_final.json`, `runs/gpt52_level{2,3}_failed/results.json`

#### gpt-5.2-pro

| Level | Failed | Compile | Val Mismatch | Timeout | API Rate Limit | Other |
|:-----:|:------:|:-------:|:------------:|:-------:|:--------------:|:-----:|
| 1 | 68 | 0 | 0 | 68 (100%) | 0 | 0 |
| 2 | 94 | 0 | 0 | 77 (81.9%) | 17 (18.1%) | 0 |
| 3 | 50 | 0 | 0 | 0 | 50 (100%) | 0 |

> Data sources: `runs/gpt-5.2-pro_failed/results_final.json`, `runs/gpt52pro_level{2,3}_failed/results.json`

### 5.3 Error Profile Comparison (Summary)

| Model | Primary Failure Mode | Secondary | Interpretation |
|-------|---------------------|-----------|----------------|
| gpt-4o-mini | Compile (71-82%) | Value mismatch (6-22%) | Cannot reliably produce valid Triton syntax |
| gpt-4o | Compile (38-51%) | Value mismatch (37-56%) | Better syntax, but semantic correctness remains hard |
| gpt-5.2 | Value mismatch (59-72%) | Compile (7-26%) | Strong syntax generation; semantic correctness is bottleneck |
| gpt-5.2-pro | Timeout/Rate limit (100%) | — | Reasoning overhead makes model impractical for this task |

### 5.4 Key Error Analysis Insights

1. **Compile error rate inversely correlates with model capability:** gpt-4o-mini (71-82%) > gpt-4o (38-51%) > gpt-5.2 (7-26%). More capable models produce syntactically valid Triton code more reliably.
2. **Value mismatch becomes the dominant failure mode** for more capable models: gpt-5.2 fails primarily on logical correctness (59-72%), suggesting that Triton-specific semantic understanding is the current bottleneck.
3. **gpt-5.2's compile error rate on Level 1 (6.9%) is remarkably low,** suggesting near-fluency in Triton syntax.
4. **Higher-level problems shift error distribution:** On Level 3, even gpt-4o's failures shift toward value mismatch (56%), indicating that harder problems produce fewer trivial syntax failures and more deep logical errors.

---

## 6. Speedup Analysis

### 6.1 Speedup Distribution of Solved Problems

#### gpt-4o Level 1 (19 solved)

| Problem | Speedup | Problem | Speedup |
|:-------:|:-------:|:-------:|:-------:|
| 2 | 1.52x | 20 | 0.99x |
| 5 | 0.96x | 21 | 0.99x |
| 6 | 0.09x | 22 | 0.99x |
| 7 | 1.06x | 25 | 2.42x |
| 8 | 1.11x | 27 | 1.00x |
| 9 | 1.04x | 29 | 0.99x |
| 12 | **11.86x** | 30 | **3.40x** |
| 13 | 1.46x | 31 | 1.00x |
| 15 | 1.10x | 32 | 1.00x |
| 19 | 0.99x | | |

> Data source: `runs/gpt4o_level1_v2/results.json` — `per_problem[].speedup` for correct entries

#### gpt-4o Level 2 (6 solved)

| Problem | Speedup |
|:-------:|:-------:|
| 9 | 1.63x |
| 18 | 2.09x |
| 38 | 2.85x |
| 49 | 6.54x |
| 70 | **19.10x** |
| 88 | 2.32x |

> Data source: `runs/gpt4o_level2/results.json`

#### gpt-5.2 Level 1 on failures (23 solved)

| Problem | Speedup | Problem | Speedup |
|:-------:|:-------:|:-------:|:-------:|
| 1 | 2.34x | 40 | 1.66x |
| 3 | 1.81x | 41 | 2.90x |
| 10 | 1.59x | 42 | 0.88x |
| 11 | 0.51x | 43 | 0.65x |
| 16 | 1.64x | 44 | 1.35x |
| 17 | 1.71x | 89 | 1.15x |
| 18 | 1.42x | 95 | 1.55x |
| 23 | 0.27x | 98 | 2.74x |
| 26 | 0.99x | 99 | **4.05x** |
| 28 | 0.83x | 100 | 2.52x |
| 35 | 0.67x | | |
| 36 | 1.21x | | |
| 37 | 1.00x | | |

> Data source: `runs/gpt-5.2_failed/results_final.json`

#### gpt-5.2 Level 2 on failures (9 solved)

| Problem | Speedup |
|:-------:|:-------:|
| 23 | 0.02x |
| 40 | 3.17x |
| 45 | 0.42x |
| 51 | 1.18x |
| 64 | 0.11x |
| 66 | 1.32x |
| 80 | 2.15x |
| 83 | 0.05x |
| 99 | 2.00x |

> Data source: `runs/gpt52_level2_failed/results.json`

#### gpt-5.2 Level 3 (4 solved)

| Problem | Speedup |
|:-------:|:-------:|
| 16 | **84.03x** |
| 38 | 1.00x |
| 43 | 4.26x |
| 46 | 0.63x |

> Data source: `runs/gpt52_level3_failed/results.json`

#### gpt-5.2-pro Level 1 on failures (13 solved)

| Problem | Speedup | Problem | Speedup |
|:-------:|:-------:|:-------:|:-------:|
| 1 | 1.78x | 37 | 1.00x |
| 3 | 1.00x | 41 | 1.98x |
| 10 | 0.99x | 44 | 1.31x |
| 17 | 1.68x | 88 | **7.60x** |
| 18 | 2.30x | 98 | **5.32x** |
| 26 | 1.00x | 100 | **7.35x** |
| 28 | 0.99x | | |

> Data source: `runs/gpt-5.2-pro_failed/results_final.json`

#### gpt-4o-mini Level 1 (15 solved)

| Problem | Speedup | Problem | Speedup |
|:-------:|:-------:|:-------:|:-------:|
| 8 | 1.18x | 23 | 1.89x |
| 9 | 1.03x | 25 | 2.42x |
| 14 | 0.001x | 27 | 0.99x |
| 15 | 0.73x | 29 | 0.99x |
| 19 | 0.99x | 30 | **3.40x** |
| 20 | 0.99x | 31 | 0.99x |
| 21 | 0.99x | 32 | 0.99x |
| 22 | 1.00x | | |

> Data source: `runs/gpt4o_mini_level1/results.json`

#### gpt-4o-mini Level 2 (1 solved)

| Problem | Speedup |
|:-------:|:-------:|
| 9 | **25.45x** |

> Data source: `runs/gpt4o_mini_level2/results.json`

### 6.2 Top 10 Highest Speedups (All Experiments)

| Rank | Problem | Level | Model | Speedup |
|:----:|:-------:|:-----:|-------|:-------:|
| 1 | L3-P16 | 3 | gpt-5.2 | **84.03x** |
| 2 | L2-P9 | 2 | gpt-4o-mini | **25.45x** |
| 3 | L2-P70 | 2 | gpt-4o | **19.10x** |
| 4 | L1-P12 | 1 | gpt-4o | **11.86x** |
| 5 | L1-P88 | 1 | gpt-5.2-pro | **7.60x** |
| 6 | L1-P100 | 1 | gpt-5.2-pro | **7.35x** |
| 7 | L2-P49 | 2 | gpt-4o | **6.54x** |
| 8 | L1-P98 | 1 | gpt-5.2-pro | **5.32x** |
| 9 | L1-P99 | 1 | gpt-5.2 | **4.05x** |
| 10 | L3-P43 | 3 | gpt-5.2 | **4.26x** |

### 6.3 Speedup Tier Analysis

| Speedup Tier | gpt-4o-mini | gpt-4o | gpt-5.2 | gpt-5.2-pro |
|:------------:|:-----------:|:------:|:-------:|:-----------:|
| < 1.0x (correct but slower) | 9 | 9 | 11 | 3 |
| 1.0x - 2.0x | 5 | 7 | 17 | 7 |
| 2.0x - 5.0x | 1 | 3 | 5 | 3 |
| 5.0x - 10.0x | 0 | 0 | 0 | 2 |
| > 10.0x | 0 | 1 | 1 | 0 |
| **Fast-1 (>= 1.0x)** | 7/16 (43.8%) | 12/25 (48.0%) | 25/36 (69.4%) | 10/13 (76.9%) |
| **Fast-2 (>= 2.0x)** | 1/16 (6.3%) | 4/25 (16.0%) | 5/36 (13.9%) | 3/13 (23.1%) |

---

## 7. Per-Problem Detailed Results

### 7.1 Level 1 — All Models Cross-Reference

| PID | gpt-4o-mini | gpt-4o | gpt-5.2 | gpt-5.2-pro | Best Speedup |
|:---:|:----------:|:------:|:-------:|:-----------:|:------------:|
| 1 | - | - | 2.34x | 1.78x | 2.34x (5.2) |
| 2 | - | 1.52x | N/A | N/A | 1.52x (4o) |
| 3 | - | - | 1.81x | 1.00x | 1.81x (5.2) |
| 5 | - | 0.96x | N/A | N/A | 0.96x (4o) |
| 6 | - | 0.09x | N/A | N/A | 0.09x (4o) |
| 7 | - | 1.06x | N/A | N/A | 1.06x (4o) |
| 8 | 1.18x | 1.11x | N/A | N/A | 1.18x (mini) |
| 9 | 1.03x | 1.04x | N/A | N/A | 1.04x (4o) |
| 10 | - | - | 1.59x | 0.99x | 1.59x (5.2) |
| 11 | - | - | 0.51x | - | 0.51x (5.2) |
| 12 | - | 11.86x | N/A | N/A | 11.86x (4o) |
| 13 | - | 1.46x | N/A | N/A | 1.46x (4o) |
| 14 | 0.001x | - | - | - | 0.001x (mini) |
| 15 | 0.73x | 1.10x | N/A | N/A | 1.10x (4o) |
| 16 | - | - | 1.64x | - | 1.64x (5.2) |
| 17 | - | - | 1.71x | 1.68x | 1.71x (5.2) |
| 18 | - | - | 1.42x | 2.30x | 2.30x (pro) |
| 19 | 0.99x | 0.99x | N/A | N/A | 0.99x |
| 20 | 0.99x | 0.99x | N/A | N/A | 0.99x |
| 21 | 0.99x | 0.99x | N/A | N/A | 0.99x |
| 22 | 1.00x | 0.99x | N/A | N/A | 1.00x |
| 23 | 1.89x | - | 0.27x | - | 1.89x (mini) |
| 25 | 2.42x | 2.42x | N/A | N/A | 2.42x |
| 26 | - | - | 0.99x | 1.00x | 1.00x (pro) |
| 27 | 0.99x | 1.00x | N/A | N/A | 1.00x (4o) |
| 28 | - | - | 0.83x | 0.99x | 0.99x (pro) |
| 29 | 0.99x | 0.99x | N/A | N/A | 0.99x |
| 30 | 3.40x | 3.40x | N/A | N/A | 3.40x |
| 31 | 0.99x | 1.00x | N/A | N/A | 1.00x (4o) |
| 32 | 0.99x | 1.00x | N/A | N/A | 1.00x (4o) |
| 35 | - | - | 0.67x | - | 0.67x (5.2) |
| 36 | - | - | 1.21x | - | 1.21x (5.2) |
| 37 | - | - | 1.00x | 1.00x | 1.00x |
| 40 | - | - | 1.66x | - | 1.66x (5.2) |
| 41 | - | - | 2.90x | 1.98x | 2.90x (5.2) |
| 42 | - | - | 0.88x | - | 0.88x (5.2) |
| 43 | - | - | 0.65x | - | 0.65x (5.2) |
| 44 | - | - | 1.35x | 1.31x | 1.35x (5.2) |
| 88 | - | - | - | 7.60x | 7.60x (pro) |
| 89 | - | - | 1.15x | - | 1.15x (5.2) |
| 95 | - | - | 1.55x | - | 1.55x (5.2) |
| 98 | - | - | 2.74x | 5.32x | 5.32x (pro) |
| 99 | - | - | 4.05x | - | 4.05x (5.2) |
| 100 | - | - | 2.52x | 7.35x | 7.35x (pro) |

> `-` = failed, `N/A` = not attempted (already solved by gpt-4o)

---

## 8. Key Findings

### Finding 1: Multi-Model Cascading is Highly Effective
The cascading strategy (gpt-4o → gpt-5.2) achieves **2.44x improvement** over single-model evaluation (25 → 61 out of 250 problems). This suggests that different models find qualitatively different solutions, making ensemble approaches valuable.

### Finding 2: Reasoning Models are Inefficient for Code Generation
gpt-5.2-pro, despite being a more expensive reasoning model, solves **fewer problems** than gpt-5.2 (13 vs. 23 on L1 failures) and adds only **1 unique problem** to the combined set. The reasoning overhead leads to excessive timeouts (100% on L1, 82% on L2) and API rate limits (100% on L3).

### Finding 3: Compile Error Rate is a Proxy for Model Capability
The compile error rate decreases monotonically with model capability: gpt-4o-mini (71-82%) > gpt-4o (38-51%) > gpt-5.2 (7-26%). This suggests that Triton syntax fluency can serve as a proxy measure for LLM coding capability in domain-specific languages.

### Finding 4: Value Mismatch is the Hard Bottleneck
For the most capable model (gpt-5.2), **59-72% of failures are value mismatches** — the model produces syntactically valid, compilable code that produces incorrect results. This represents a fundamental challenge in LLM-based kernel generation that cannot be addressed through syntax feedback alone.

### Finding 5: Stochastic Diversity Outperforms Iteration Depth
The ablation study shows that running 4 configurations (iter=1/3/5/10) and taking the union (28 solved) outperforms any single configuration by 27.3%. Each iteration count finds **uniquely solvable problems**, suggesting a best-of-N independent runs strategy may be more effective than deep iterative refinement.

### Finding 6: Level 3 Represents a Qualitative Difficulty Barrier
Level 3 problems are unsolvable by gpt-4o and gpt-4o-mini (0/50 each), with only gpt-5.2 achieving 4 solutions. The exceptional speedup on L3-P16 (84.03x) suggests that when LLMs do solve complex problems, the generated Triton code can dramatically outperform naive PyTorch implementations.

### Finding 7: Error Profile Shifts Across Iteration Count
More iterations reduce compile errors (59.3% → 32.1%) but increase value mismatches (24.7% → 53.8%). The iterative refinement loop successfully fixes syntactic issues via error feedback, but **exposes deeper logical errors** that the model cannot self-correct.

### Finding 8: High Speedup Outliers Suggest Significant Optimization Potential
The top speedups (84x, 25x, 19x, 12x) are orders of magnitude above the median, indicating that for specific computation patterns, LLM-generated Triton kernels can achieve substantial performance gains over PyTorch reference implementations.

---

## 9. Data Source Reference

### 9.1 Primary Results Files

All paths are relative to the project root (`/home/jemin/development/llm_kernel/`).

| Experiment | Results File | Key Fields |
|-----------|-------------|------------|
| gpt-4o Level 1 | `runs/gpt4o_level1_v2/results.json` | per_problem[].{correct, speedup, compile_error, runtime_error, error_message} |
| gpt-4o Level 2 | `runs/gpt4o_level2/results.json` | same structure |
| gpt-4o Level 3 | `runs/gpt4o_level3/results.json` | same structure |
| gpt-5.2 Level 1 | `runs/gpt-5.2_failed/results_final.json` | same structure (also results_partial.json, identical) |
| gpt-5.2 Level 2 | `runs/gpt52_level2_failed/results.json` | same structure |
| gpt-5.2 Level 3 | `runs/gpt52_level3_failed/results.json` | same structure |
| gpt-5.2-pro Level 1 | `runs/gpt-5.2-pro_failed/results_final.json` | same structure (also results_partial.json, identical) |
| gpt-5.2-pro Level 2 | `runs/gpt52pro_level2_failed/results.json` | same structure |
| gpt-5.2-pro Level 3 | `runs/gpt52pro_level3_failed/results.json` | same structure |
| gpt-4o-mini Level 1 | `runs/gpt4o_mini_level1/results.json` | same structure |
| gpt-4o-mini Level 2 | `runs/gpt4o_mini_level2/results.json` | same structure |
| gpt-4o-mini Level 3 | `runs/gpt4o_mini_level3/results.json` | same structure |
| Ablation iter=1 | `runs/ablation_iter1/results.json` | same structure |
| Ablation iter=3 | `runs/ablation_iter3/results.json` | same structure |
| Ablation iter=10 | `runs/ablation_iter10/results.json` | same structure |
| Ablation iter=5 | `runs/gpt4o_level1_v2/results.json` | (same as gpt-4o L1 baseline) |

### 9.2 Results JSON Schema

```json
{
  "run_name": "string",
  "model": "string",
  "level": "int",
  "total_problems": "int",
  "solved_count": "int",
  "per_problem": [
    {
      "problem_id": "int",
      "level": "int",
      "correct": "bool",
      "speedup": "float",
      "compile_error": "bool",
      "runtime_error": "bool",
      "error_message": "string | null"
    }
  ]
}
```

### 9.3 Verification Instructions

To manually verify any number in this report:

```python
import json

# Example: Verify gpt-4o Level 1 solved count
with open("runs/gpt4o_level1_v2/results.json") as f:
    data = json.load(f)

solved = [r for r in data["per_problem"] if r["correct"]]
print(f"Solved: {len(solved)}/{len(data['per_problem'])}")
for r in solved:
    print(f"  P{r['problem_id']}: speedup={r['speedup']:.4f}x")
```

```python
# Example: Verify union of gpt-4o + gpt-5.2 for Level 1
import json

gpt4o = json.load(open("runs/gpt4o_level1_v2/results.json"))
gpt52 = json.load(open("runs/gpt-5.2_failed/results_final.json"))

gpt4o_solved = {r["problem_id"] for r in gpt4o["per_problem"] if r["correct"]}
gpt52_solved = {r["problem_id"] for r in gpt52["per_problem"] if r["correct"]}
union = gpt4o_solved | gpt52_solved
print(f"gpt-4o: {len(gpt4o_solved)}, gpt-5.2: {len(gpt52_solved)}, union: {len(union)}")
```

```python
# Example: Verify error distribution
import json
from collections import Counter

data = json.load(open("runs/gpt4o_level1_v2/results.json"))
failed = [r for r in data["per_problem"] if not r["correct"]]

errors = Counter()
for r in failed:
    msg = r.get("error_message", "") or ""
    if r.get("compile_error"):
        errors["compile"] += 1
    elif "mismatch" in msg.lower() or "values" in msg.lower() or "incorrect" in msg.lower():
        errors["value_mismatch"] += 1
    elif "cuda" in msg.lower() or "illegal" in msg.lower():
        errors["cuda"] += 1
    elif "timeout" in msg.lower():
        errors["timeout"] += 1
    else:
        errors["other"] += 1

print(f"Total failed: {len(failed)}")
for k, v in errors.most_common():
    print(f"  {k}: {v} ({v/len(failed)*100:.1f}%)")
```

### 9.4 Individual Problem Results

Each problem also has its own results directory:

```
runs/{run_prefix}_p{problem_id}/results.json
```

For example:
- `runs/gpt4o_level2_p70/results.json` — gpt-4o Level 2, Problem 70 (19.10x speedup)
- `runs/gpt52_level3_failed_p16/results.json` — gpt-5.2 Level 3, Problem 16 (84.03x speedup)
- `runs/gpt-5.2-pro_p88/results.json` — gpt-5.2-pro Level 1, Problem 88 (7.60x speedup)

### 9.5 Experiment Runner Scripts

| Script | Purpose |
|--------|---------|
| `run_level2_level3.py` | Level 2/3 gpt-4o + gpt-5.2 cascading |
| `run_ablation_iterations.py` | Iteration ablation study (1/3/10) |
| `run_4o_mini.py` | gpt-4o-mini baseline (all levels) |
| `run_5.2pro_level2_3.py` | gpt-5.2-pro on Level 2/3 failures |

### 9.6 Pipeline Source Code

| Module | Path | Role |
|--------|------|------|
| CLI | `src/llm_kernel/cli.py` | Run orchestration, iterative refinement |
| LLM Client | `src/llm_kernel/client/llm_client.py` | API calls (Chat/Completions/Responses) |
| Prompt Builder | `src/llm_kernel/client/prompt_builder.py` | System prompt + GPU hints |
| Response Parser | `src/llm_kernel/client/response_parser.py` | Code extraction, cleanup |
| Evaluator | `src/llm_kernel/bench/runner.py` | Subprocess-isolated CUDA evaluation |
| GPU Knowledge | `src/llm_kernel/knowledge/` | GPU specs, Triton patterns |

---

## Appendix A: Experimental Notes

### A.1 gpt-5.2-pro Level 3 Caveat
The gpt-5.2-pro Level 3 results show 0/50 solved, but **all 50 failures are API rate-limit errors (HTTP 429)**, not genuine model failures. This means gpt-5.2-pro was never actually evaluated on Level 3 problems. The Level 2 results (0/94 with 82% timeout, 18% rate-limit) are more representative but still limited by infrastructure constraints.

### A.2 CUDA Error Isolation
Each kernel evaluation runs in a separate subprocess to prevent CUDA context corruption from cascading to subsequent problems. This is critical for reliable benchmarking — see `src/llm_kernel/bench/runner.py`.

### A.3 Evaluation Environment
- **GPU:** NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Compute Capability:** sm_120
- **CUDA:** 12.x
- **Triton:** 3.5.1
- **PyTorch:** 2.x
- **Python:** 3.12
- **Conda Environment:** llm_kernel
