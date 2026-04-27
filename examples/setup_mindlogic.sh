#!/bin/bash
# Mindlogic AI Gateway 환경 설정
# 사용법: source examples/setup_mindlogic.sh

# ============================================================
# 1. API 키 설정 (한 번만 설정하면 모든 SDK에서 사용 가능)
# ============================================================
export MINDLOGIC_API_KEY="0Cg54ZNhkO30fxlD5zjEnymAYCD2XhRE"

# ============================================================
# 2. OpenAI SDK용 환경변수
#    → gpt-5.4, claude-sonnet-4-6, gemini 등 모든 모델 사용 가능
# ============================================================
export OPENAI_API_KEY="$MINDLOGIC_API_KEY"
export OPENAI_BASE_URL="https://factchat-cloud.mindlogic.ai/v1/gateway"

# ============================================================
# 3. Claude Code CLI 연동
#    → claude 명령어가 Mindlogic Gateway를 통해 동작
# ============================================================
export ANTHROPIC_BASE_URL="https://factchat-cloud.mindlogic.ai/v1/gateway/claude"
export ANTHROPIC_AUTH_TOKEN="$MINDLOGIC_API_KEY"

# ============================================================
# 4. Anthropic SDK용 (Python에서 anthropic 패키지 사용 시)
# ============================================================
export ANTHROPIC_API_KEY="$MINDLOGIC_API_KEY"

echo "Mindlogic AI Gateway 환경 설정 완료"
echo "  OpenAI SDK:   $OPENAI_BASE_URL"
echo "  Claude Code:  $ANTHROPIC_BASE_URL"
echo ""
echo "테스트:"
echo "  python examples/mindlogic_gateway_example.py"
echo "  claude --model claude-sonnet-4-6 'Hello'"
