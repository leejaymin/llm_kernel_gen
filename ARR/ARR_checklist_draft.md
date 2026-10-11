# Responsible NLP Checklist 초안 (OpenReview 폼 입력용, 2026-10-09)

규칙: Yes → 근거 섹션 명시. No → 사유. N/A → 사유. 근거 없는 일괄 Yes·누락은 desk reject 사유.
섹션 번호는 현재 main.pdf 기준 (§3 Study Design, §4.1 RQ1, §4.2 RQ2, Limitations, Ethical Considerations, App. B Budget, App. L Variance).

## A. For every submission
| 항목 | 답 | 폼에 적을 근거 (영문) |
|---|---|---|
| A1 Limitations | Yes | Section "Limitations": single-run numbers except one repeated configuration (sd ≈ 4 points), 20-task scope of some large-budget runs, uncontrolled API serving conditions, small OOD set, timeout column semantics. |
| A2 Risks | Yes | Section "Ethical Considerations": misuse of unverified generated kernels in production; mitigated by making verification strength a required reporting axis. No human data. |

## B. Scientific artifacts (사용만 함, 신규 공개 없음)
| 항목 | 답 | 근거 |
|---|---|---|
| B1 Cite creators | Yes | Section 3 (KernelBench, TritonBench, TritonGym, FlashInfer-Bench, Qwen3-Coder, AutoTriton, Dr.Kernel, GLM) and Appendix G (AdaExplore, open agent framework, RL environment); versions in Section 3 (Triton 3.5.1/3.6, PyTorch 2.x, Python 3.12, CUDA 12.x). |
| B2 License / terms | Yes | Section "Ethical Considerations" (license sentence added 2026-10-09; 아래 "라이선스 조사 결과" 참고). |
| B3 Intended use | Yes | Section "Ethical Considerations": all artifacts are public research releases used for research evaluation only; no redistribution of modified weights or data. |
| B4 PII / offensive content | N/A | We did not collect or release any data; the benchmarks contain only PyTorch reference code and tensor shapes. |
| B5 Documentation of artifacts | N/A (or Yes) | We release no new artifact. The benchmark compositions we use are described in Section 3 (task counts per level, OOD split). |
| B6 Statistics (splits) | Yes | Section 3: KernelBench L1 100 / L2 100 / L3 50, TritonBench 166, TritonGym 139 standard + 13 OOD, FlashInfer-Bench 8; all tasks are test tasks, no training. |

## C. Computational experiments
| 항목 | 답 | 근거 |
|---|---|---|
| C1 Parameters / budget / infra | Yes | Section 3 (model sizes 8B/14B/30B, GPUs: RTX PRO 6000 96GB, H100 80GB, A6000 48GB, eight-GPU node for the RL reproduction) and Appendix B (wall-clock per sweep, 22.9 h large-budget sweep, ~32 GPU-hours for the repeated runs). |
| C2 Setup / hyperparameters | Yes | Section 3 and Appendix A (prompt, isolation, correctness tolerance 1e-2 on two inputs, median of 10 timing runs), Appendix B (budgets), Appendix L (temperature 0.2 / 0.3, timeouts 2400 s / 120 s, verification inputs 2 / 5). No hyperparameter search was performed; settings are fixed defaults. |
| C3 Descriptive statistics / variance | Yes | Section 4.1 and Appendix L (Table 13); NPU results in Appendix J are pass counts over 3 samples per cell: four repeated runs of the main configuration, mean ± sd = 0.44 ± 0.04, union/intersection; all other numbers are explicitly marked as single runs (Table 1, 5, 7 captions; Limitations). |
| C4 Packages / versions | Yes | Section 3 (Triton, PyTorch, Python, CUDA, vLLM, llama.cpp), Appendix A, G and J (vendor NPU SDKs: rebel-compiler 0.10.x Triton dialect; furiosa-opt-std 0.8.1 Rust DSL) (modifications to reused code, e.g. generation-config fix, chunked comparison). |

## D. Human annotators
| D1–D5 | N/A | No human annotators, participants, or crowd workers. |

## E. AI assistants
| 항목 | 답(제안) | 근거 |
|---|---|---|
| E1 AI assistant use | **Yes** | Draft wording: "An AI coding assistant was used to help write and debug the evaluation harness code and to edit the English of the manuscript. All experimental design, measurements, and claims were made and verified by the authors." → 익명 제출본에는 Acknowledgements를 넣을 수 없으므로 폼의 E1 설명란에만 기재하고, 카메라레디에서 Acknowledgements에 추가. |

E1은 저자가 최종 결정해야 함. 이 저장소의 하니스 코드와 원고 상당 부분이 AI 보조로 작성되었으므로 "No"는 체크리스트 허위 기재(desk reject 사유)가 될 수 있음.

## 라이선스 조사 결과 (B2용)
(아래는 2026-10-09 조사; 제출 전 각 저장소 LICENSE 재확인 권장)
| Artifact | 확인 URL | 라이선스 | 비고 |
|---|---|---|---|
| KernelBench 코드 | github.com/ScalingIntelligence/KernelBench | MIT | LICENSE.md |
| KernelBench 데이터셋 (HF) | huggingface.co/datasets/ScalingIntelligence/KernelBench | **미선언** | 카드에 license 필드 없음 |
| TritonBench | github.com/thunlp/TritonBench | Apache-2.0 | |
| AutoTriton 코드 | github.com/AI9Stars/AutoTriton | **라이선스 파일 없음** | README만 |
| AutoTriton 모델 (HF) | huggingface.co/ai9stars/AutoTriton | Apache-2.0 | |
| Dr.Kernel 코드 (KernelGYM) | github.com/hkust-nlp/KernelGYM | Apache-2.0 (README 명시만) | LICENSE 파일 없음 |
| Dr.Kernel-14B 모델 (HF) | huggingface.co/hkust-nlp/drkernel-14b | **미선언** | base Qwen3-14B-Base |
| TritonGym | github.com/yil384/TritonGym | MIT | 공식 repo 여부 미확인 |
| FlashInfer-Bench | github.com/flashinfer-ai/flashinfer-bench | Apache-2.0 | |
| AdaExplore | github.com/StigLidu/AdaExplore | MIT | |
| Qwen3-Coder-30B-A3B-Instruct | huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct | Apache-2.0 | |
| GLM-5.2 원본 가중치 | huggingface.co/zai-org/GLM-5.2 | MIT | GGUF 양자화본 카드는 미확인 |

→ Ethical Considerations에 위 내용을 요약한 라이선스 문단 추가(10/09). 미선언 항목은 "no explicit license, evaluation only"로 기술.


## 참고문헌 검증 결과 (10/09, 인용 42건)
- 수정함(egbib.bib, 백업 egbib.bib.bak_20261009): **KernelBench 저자 목록 오류(Ouyang 외 전원 불일치) → ICML 2025 정식 저자·제목으로 교체**, **TritonGym "Anonymous/ICLR under review" → ICML 2026 실제 저자**, CUDA-L1 저자, CuTeGen 저자 2명 누락, KernelEvolve 연도 2026→2025, EGG ICML 2026 미확인 → arXiv preprint로.
- 유지: COLT(arXiv v1·v2 제목 동일 확인), 나머지 35건 OK. kirk2016 교재는 미검색.
