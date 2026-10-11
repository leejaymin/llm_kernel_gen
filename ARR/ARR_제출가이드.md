# ACL ARR 제출 가이드 (2026-09-05 기준)

출처: https://aclrollingreview.org/cfp , /authors , /authorchecklist , /dates , /responsibleNLPresearch

## 1. 이 디렉토리

| 파일 | 내용 |
|---|---|
| `main.tex` | 본문 8쪽 + Limitations + Ethical Considerations + References + Appendix A–L (한 PDF) |
| `appendix.tex` | 구 supplementary.tex (A–K) + 본문에서 옮긴 Practical Notes (K) |
| `fig_overview.tex` | Figure 1 (TikZ, 이미지 파일 없음) |
| `egbib.bib` | 참고문헌 |
| `acl.sty`, `acl_natbib.bst` | 공식 스타일 (acl-org/acl-style-files, 2026-06-29 커밋). **수정 금지** |
| `Makefile` | `make` 한 번이면 빌드. `make clean`으로 보조파일 정리 |

CGO27 대비 변경점: `\cite`→`\citep`/`\citet`(저자-연도), Threats to Validity → `\section*{Limitations}`(ARR 필수 제목), Practical Notes → 부록, Ethical Considerations 신규(짧게), 부록을 본문 PDF에 병합.

## 2. 형식 규정 (desk-reject 항목)

- 장문 8쪽(본문+표+그림) + 참고문헌/Limitations/Ethics/부록 무제한. **현재 본문 8쪽, Conclusion·Limitations가 8쪽에서 끝남.** 본문에 한 문단이라도 추가하면 9쪽으로 넘어감.
- 제목이 정확히 `Limitations`인 절 필수. 없으면 desk reject.
- `\usepackage[review]{acl}`: 익명·줄번호·쪽번호 자동. 스타일 파일 수정 금지.
- 부록은 참고문헌 뒤, 2단. 별도 supplementary PDF 아님.
- 본문은 자기완결. 리뷰어는 부록을 읽을 의무 없음. Related Work를 부록으로 빼면 desk reject.
- 익명화: 저자 식별 가능한 자기인용 금지, Acknowledgements 금지, Dropbox 등 파일호스팅 링크 금지. 코드는 Anonymous GitHub 등.
- Responsible NLP Checklist는 OpenReview 폼 안에서 작성. 근거 없는 일괄 Yes, 누락 → desk reject.

## 3. 일정 (2026)

| 사이클 | 제출 | 저자 리뷰어 등록 | 리뷰 공개 | 저자 응답 | 메타리뷰 | 커밋 마감 | 연계 학회 |
|---|---|---|---|---|---|---|---|
| 8월 | 8/3 (지남) | 8/5 | 9/7 | 9/14–24 | 10/8 | 10/11 | EACL 2027 |
| **10월** | **10/12** | 10/14 | 11/16 (리뷰 마감) | 11/24–11/30 | 12/17 | 12/23 (사이클 종료 12/20) | NAACL 2027, COLING 2027 |

(2026-10-09 /dates 재확인. 2026년 10월 사이클부터: 전 저자 OpenReview 프로필에 ORCID·소속 이력·이메일·COI 완비, 마감 후 1주 내 미완 시 desk reject. 서비스(리뷰) 참여자 1인당 최대 2편, 마감 후 48시간 내 등록 폼 제출.)

오늘부터 10/12까지 5주. 실험 추가는 10/5 이전 완료가 현실적.

## 4. OpenReview 제출 절차

1. **전 저자 OpenReview 프로필 완성** (제출 전): 소속, 인증 이메일, DBLP·Semantic Scholar·ACL Anthology 링크. 제출 후 저자 목록·순서 변경 불가.
2. ARR 사이클 페이지 접속: `https://openreview.net/group?id=aclweb.org/ACL/ARR/2026/October` (사이클 열리면 활성화) → "ARR Submission".
3. 폼 입력:
   - 제목, 초록, 저자, PDF 1개 (본문+부록), 장문/단문, 연구 분야(area)
   - 선행 ARR 제출 URL (재제출이면 필수) + 수정 설명 PDF
   - 리뷰어 선호 (같은 리뷰어/새 리뷰어)
   - 프리프린트 상태: "비익명 프리프린트 없음"을 고르면 메타리뷰 전까지 arXiv 금지(위반 시 desk reject). arXiv 계획 있으면 반대로 선택.
   - 소프트웨어/데이터 zip (익명화, 선택)
   - Responsible NLP Checklist A–E: 각 항목에 근거 섹션 명시. C1 컴퓨팅 예산(GPU 시간·모델 파라미터), C3 분산/시드, E AI 보조 사용 공개.
4. 마감 최소 1시간 전 제출. 마감까지 수정 가능, 메타데이터(체크리스트 등)는 마감 후 48시간 편집 가능.
5. **마감 후 48시간 내 저자 리뷰어 등록 폼 제출** (2025년 5월부터 전 저자 의무). 미등록 시 desk reject. 자격: *ACL 메인/Findings 2편 + 기타 1편. 지연 리뷰 시 팀 전체가 다음 사이클 제출 불가.
6. 리뷰 공개 → 저자 응답(텍스트만, 링크 금지, 스레드당 2회 이하, "소규모 추가 실험"만 허용).
7. 메타리뷰 후 두 갈래:
   - **커밋**: 학회 커밋 폼에 OpenReview forum URL 제출 (NAACL/COLING 2027은 12/23). 동시 다중 커밋은 학회가 보통 불허.
   - **수정 재제출**: 다음 사이클에 이전 URL + 변경 요약 + 항목별 답변 PDF. 메타 1.5는 다음 사이클 불가, 1은 영구 불가.
8. 마감 48시간 이후 철회 → 다음 사이클 재제출 불가.

## 5. 이 논문 특유의 주의점

- **스코프 리스크**: ARR CFP는 "computational language processing". LLM 코드생성은 관례상 수용되나, 리뷰어가 시스템 논문으로 볼 수 있음. area는 "Resources and Evaluation" 또는 "NLP Applications" 계열 권장. 폼의 실제 area 목록에서 확인.
- 체크리스트 C3(오차막대): **완료(10/09)**. §4.1 + 부록 L(표 13) 4회 반복 0.44±0.04. 나머지 수치는 단일 실행임을 표 캡션·Limitations에 명시.
- 체크리스트 C1: **완료(10/09)**. 부록 B에 반복 실행 GPU 시간(약 32 GPU-hours) 추가.
- 체크리스트 초안: `ARR_checklist_draft.md`.
- 오픈소스 재현물(AutoTriton, Dr.Kernel, AdaExplore 등) 라이선스는 B2 항목에 기재 필요.

## 6. 실험 우선순위 (ARR 기준, 10/12 마감)

| 순위 | 실험 | 자원 | 근거 |
|---|---|---|---|
| ① | RQ1 상위 모델 3~5회 반복 → 표준편차 | API, 저렴 | 체크리스트 C3 직접 요구. Finding 4의 "합집합 28 vs 단일 22"가 샘플링 분산인지 검증 |
| ② | 대형 예산 L1 전수 완주 (AT-8B, Qwen) | H100 임대 | Finding 5 (0.95)가 20문제뿐. Limitations 첫 문장 제거 |
| ③ | 신규 오픈 모델 (Qwen3.8 등) 소예산 L1 | H100 임대 | Finding 1 검증. NLP 리뷰어가 가장 먼저 보는 모델 커버리지 |
| ④ | 메모리 헤드룸 곡선 (0.90/0.75/0.60/0.45) | H100 임대 | Finding 2를 일화에서 곡선으로 |
| ⑤ | torch.compile 베이스라인 | 로컬, 비용 0 | speedup의 의미. ACL 리뷰어에게는 CGO보다 덜 치명적 |
| ⑥ | opus-5 L2 재개 (P7–P20) | API 2만 크레딧 | 결과 0/6이라 논문 기여 낮음. ①에 크레딧 우선 배정 |
| ⑦ | Triton 버전 스윕, 컴파일 단계별 오류 분해 | H100 / 로컬 | 컴파일러 학회용. ARR에서는 후순위 |

본문이 이미 8쪽 상한이므로 새 결과는 기존 표의 열 추가(±표준편차, 새 모델 행) 형태로만 넣고 서술은 부록으로.


## 7. 2026-10-09 규정 적합성 점검 결과 (cfp + authorchecklist 기준)

| 항목 | 규정 | 상태 |
|---|---|---|
| 분량 | 장문 본문 8쪽 + Limitations/Ethics/참고문헌/부록 무제한 | **적합**. Conclusion이 8쪽에서 끝남, Limitations는 9쪽 시작. 총 19쪽 (10/09 NPU 부록 추가 후) |
| Limitations 절 | 제목 정확히 "Limitations", 결론 뒤·참고문헌 앞, 새 결과 금지 | **적합**. 반복 실행 결과는 §4.1/부록 L에 두고 Limitations는 참조만 |
| 스타일 파일 | 공식 acl.sty/acl_natbib.bst 무수정 | **적합**. md5가 acl-org/acl-style-files master와 동일 (10/09 확인) |
| review 옵션 | 익명 저자 블록, 줄번호, 쪽번호 | **적합**. `\usepackage[review]{acl}` |
| 부록 형식 | 2단, 본문 PDF에 포함 | **적합** |
| 익명성 | 저자명·소속·감사의 글·Dropbox 링크·식별 가능한 자기인용 금지 | **적합**. tex 전체 grep: 이메일·소속·github 개인 저장소·"our previous" 없음. PDF 메타데이터 Author 공란 |
| Overfull / 미정의 참조 | — | 0 / 0 |
| Responsible NLP Checklist | 폼에서 A–E 작성, Yes는 섹션 명시 | 초안 작성(`ARR_checklist_draft.md`). **E1(AI 보조) 저자 결정 필요** |
| 다중 제출 | 타 학회·저널 동시 심사 금지 | **저자 확인 필요**: AAAI27/CGO27 디렉토리의 버전이 현재 어디에도 under review가 아니어야 함 |
| 프리프린트 | 익명 기간 없음. "비익명 프리프린트 없음" 선택 시 위반하면 desk reject | 폼에서 선택 |
| OpenReview 프로필 | 전 저자 ORCID·소속 이력·이메일·COI, 마감 후 1주 내 | **저자 작업** |
| 리뷰어/서비스 등록 | 마감 후 48h, 1인 최대 2편 | **저자 작업** (10/14) |
| 참고문헌 정확성 | 인용 논문 실재·메타데이터 일치 | 42건 전수 대조 완료. 6건 수정(KernelBench 저자 전면 오류, TritonGym Anonymous→실제 저자 등). 상세: `ARR_checklist_draft.md` |
| B2 라이선스 | 사용 artifact 라이선스 명시 | **적합**. Ethical Considerations에 12종 라이선스 문단 추가 (미선언 3종은 그대로 명시) |


## 8. OpenReview 폼 선택안 (2026-10-09, 실제 폼 PDF 기준)

**마감 정정**: 폼 표기 Submission Deadline = **Oct 13 2026 11:59 AM UTC = 10/13(화) 20:59 KST**. 메타데이터는 마감 후 48h 편집 가능(PDF·저자 목록 제외).

| 필드 | 선택 | 근거 |
|---|---|---|
| Paper Type | Long | 본문 8쪽 |
| Research Area | **NLP and Code Models** | 키워드 "language-to-code generation"이 정확히 해당. 차선 Resources and Evaluation(benchmarking) |
| Research Area Keywords | language-to-code generation, benchmarking | areas 페이지 문구 그대로 |
| Contribution Types | **NLP engineering experiment** (1차), Reproduction study, Model analysis & interpretability | 실증 연구 + 공개 artifact 재현 |
| Keywords | LLM code generation, GPU kernel generation, Triton, KernelBench, evaluation methodology, test-time compute, reproducibility | |
| Languages Studied | 비워둠(자연어 아님) — 필수 아님 | |
| Previous URL / Revisions PDF / Reassignment | 공란 / "This is not a resubmission" ×2 | 첫 ARR 제출 |
| Concurrent Submissions | 다른 곳 심사 중 버전이 없으면 공란. 있으면 반드시 기재 | AAAI27/CGO27 상태 확인 |
| Software / Data | 선택. 올리려면 익명화된 zip(≤200MB) | |
| Preprint (anonymous) | yes 권장 | ARR 익명 프리프린트, 부담 없음 |
| Preprint Status | arXiv 계획 있으면 "We plan/considering…"; 없으면 마지막(binding) 선택 — **binding 선택 후 arXiv 올리면 desk reject** | |
| Preferred Venue | NAACL 또는 COLING (10월 사이클 커밋 대상) | |
| Visa / Country | 한국 → 미국 NAACL이면 yes, KR | |
| Consent ×2, Author Checklist, License | 모두 동의/yes | |
| Service Contributor | **반드시 지정**. 미지정 시 리뷰 보장 없음(추첨). 자격: *ACL 메인/Findings 등 2편(박사) — OpenReview 자격 검사기로 확인. 마감 후 48h 내 등록 폼 | sustainable reviewer policy |
