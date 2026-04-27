#!/usr/bin/env python3
"""
Mindlogic AI Gateway - 통합 API 예제
=====================================
OpenAI, Anthropic(Claude), Google(Gemini) 모델을 단일 API 키로 사용하는 예제.

사전 설정:
    pip install openai anthropic
    export MINDLOGIC_API_KEY="your-api-key-here"

문서: https://docs.mindlogic.ai/docs/jbnu/gateway/getting-started/authentication
"""

import os
import asyncio
from openai import OpenAI

# ============================================================
# 설정
# ============================================================
API_KEY = os.environ.get("MINDLOGIC_API_KEY", "YOUR_API_KEY")
BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"

# OpenAI SDK로 모든 제공사 모델 사용 가능 (통합 엔드포인트)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# 1. 기본 사용: 여러 모델로 동일한 질문하기
# ============================================================
def basic_multi_model_example():
    """OpenAI, Anthropic, Google 모델에 동일한 질문을 보내고 응답 비교."""

    models = [
        ("gpt-5.4-mini", "OpenAI"),
        ("claude-sonnet-4-6", "Anthropic"),
        ("gemini-3.1-pro-preview", "Google"),
    ]

    question = "Python에서 리스트 컴프리헨션과 map() 함수의 성능 차이를 간단히 설명해 주세요."

    print("=" * 70)
    print("1. 다중 모델 비교 (동일 질문)")
    print("=" * 70)

    for model_id, provider in models:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": question}],
                max_tokens=300,
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            tokens = response.usage
            print(f"\n[{provider}] {model_id}")
            print(f"  토큰: 입력={tokens.prompt_tokens}, 출력={tokens.completion_tokens}")
            print(f"  응답: {answer[:200]}...")
        except Exception as e:
            print(f"\n[{provider}] {model_id} - 에러: {e}")

    print()


# ============================================================
# 2. 스트리밍 응답
# ============================================================
def streaming_example():
    """스트리밍 모드로 실시간 응답 받기."""

    print("=" * 70)
    print("2. 스트리밍 응답 (claude-sonnet-4-6)")
    print("=" * 70)

    stream = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {"role": "user", "content": "피보나치 수열을 계산하는 3가지 방법을 Python으로 보여주세요."}
        ],
        max_tokens=500,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


# ============================================================
# 3. 시스템 프롬프트 + 멀티턴 대화
# ============================================================
def multi_turn_example():
    """시스템 프롬프트와 멀티턴 대화 예제."""

    print("=" * 70)
    print("3. 멀티턴 대화 (gpt-5.4-mini)")
    print("=" * 70)

    messages = [
        {
            "role": "system",
            "content": "당신은 GPU 프로그래밍 전문가입니다. 간결하게 답변하세요.",
        },
        {
            "role": "user",
            "content": "CUDA와 Triton의 핵심 차이점은?",
        },
    ]

    # 첫 번째 턴
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages,
        max_tokens=300,
    )
    assistant_reply = response.choices[0].message.content
    print(f"  User: {messages[-1]['content']}")
    print(f"  Assistant: {assistant_reply[:200]}...")

    # 두 번째 턴 (이전 대화 이어가기)
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "그렇다면 Triton으로 행렬 곱셈 커널을 작성할 때 주의할 점은?"})

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages,
        max_tokens=300,
    )
    print(f"\n  User: {messages[-1]['content']}")
    print(f"  Assistant: {response.choices[0].message.content[:200]}...")
    print()


# ============================================================
# 4. 코드 생성 태스크: 모델별 비교
# ============================================================
def code_generation_comparison():
    """Triton 커널 코드 생성 능력을 모델별로 비교."""

    print("=" * 70)
    print("4. 코드 생성 비교: Triton vector_add 커널")
    print("=" * 70)

    prompt = """다음 PyTorch 코드를 등가의 Triton 커널로 변환하세요.
코드만 출력하세요.

```python
import torch

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
```"""

    models = ["gpt-5.4-mini", "claude-sonnet-4-6"]

    for model_id in models:
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            code = response.choices[0].message.content
            print(f"\n--- {model_id} ---")
            print(code[:400])
            if len(code) > 400:
                print("... (truncated)")
        except Exception as e:
            print(f"\n--- {model_id} --- 에러: {e}")

    print()


# ============================================================
# 5. 사용 가능한 모델 목록 조회
# ============================================================
def list_available_models():
    """Gateway에서 사용 가능한 모델 목록 조회."""

    print("=" * 70)
    print("5. 사용 가능한 모델 목록")
    print("=" * 70)

    try:
        models = client.models.list()
        for m in models.data:
            print(f"  - {m.id}")
    except Exception as e:
        print(f"  모델 목록 조회 실패: {e}")
    print()


# ============================================================
# 6. Anthropic SDK 직접 사용 (Claude 전용 엔드포인트)
# ============================================================
def anthropic_sdk_example():
    """Anthropic SDK로 Claude 전용 엔드포인트 사용."""

    print("=" * 70)
    print("6. Anthropic SDK 직접 사용")
    print("=" * 70)

    try:
        import anthropic

        claude_client = anthropic.Anthropic(
            api_key=API_KEY,
            base_url="https://factchat-cloud.mindlogic.ai/v1/gateway/claude",
        )

        message = claude_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[
                {"role": "user", "content": "asyncio와 threading의 차이를 한 문장으로 설명해 주세요."}
            ],
        )
        print(f"  응답: {message.content[0].text}")
    except ImportError:
        print("  anthropic 패키지가 설치되지 않았습니다: pip install anthropic")
    except Exception as e:
        print(f"  에러: {e}")
    print()


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    print()
    print("Mindlogic AI Gateway - 통합 API 예제")
    print(f"Base URL: {BASE_URL}")
    print(f"API Key: {'설정됨' if API_KEY != 'YOUR_API_KEY' else '미설정 (MINDLOGIC_API_KEY 환경변수 필요)'}")
    print()

    # 모델 목록 먼저 확인
    list_available_models()

    # 기본 멀티모델 비교
    basic_multi_model_example()

    # 스트리밍
    streaming_example()

    # 멀티턴 대화
    multi_turn_example()

    # 코드 생성 비교
    code_generation_comparison()

    # Anthropic SDK 직접 사용
    anthropic_sdk_example()

    print("=" * 70)
    print("모든 예제 완료!")
    print("=" * 70)
