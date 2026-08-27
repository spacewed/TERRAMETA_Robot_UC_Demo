#!/usr/bin/env bash
# =============================================================================
# smoke-test.sh — Verify the vLLM server is running and responding
# =============================================================================
# Usage:
#   ./scripts/smoke-test.sh [BASE_URL]
#
# Default BASE_URL: http://localhost:8000
# =============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}"

echo "============================================================"
echo "  Smoke Test — vLLM Server"
echo "  Base URL: ${BASE_URL}"
echo "  Model:    ${MODEL_NAME}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Check if server is reachable
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Checking /v1/models..."
MODELS_RESPONSE=$(curl -s -f "${BASE_URL}/v1/models" 2>&1) || {
    echo "FAIL: Could not reach ${BASE_URL}/v1/models"
    echo "      Is the vLLM server running?"
    exit 1
}
echo "OK: ${MODELS_RESPONSE}" | head -c 500
echo ""

# ---------------------------------------------------------------------------
# Check model name is present
# ---------------------------------------------------------------------------
if echo "$MODELS_RESPONSE" | grep -q "$MODEL_NAME"; then
    echo "OK: Model '${MODEL_NAME}' found in /v1/models"
else
    echo "WARN: Model '${MODEL_NAME}' not found in response."
    echo "      Available models may have a different name."
fi

# ---------------------------------------------------------------------------
# Test chat completion
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Testing /v1/chat/completions (text)..."
CHAT_RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"${MODEL_NAME}"'",
        "messages": [
            {"role": "user", "content": "Reply with exactly: VLM server ready"}
        ],
        "max_tokens": 64,
        "temperature": 0.0
    }' 2>&1) || {
    echo "FAIL: Chat completion request failed."
    exit 1
}
echo "Response:"
echo "${CHAT_RESPONSE}" | python3 -m json.tool 2>/dev/null || echo "${CHAT_RESPONSE}"

# ---------------------------------------------------------------------------
# Extract assistant message
# ---------------------------------------------------------------------------
ASSISTANT_MSG=$(echo "${CHAT_RESPONSE}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data['choices'][0]['message']['content'])
except Exception as e:
    print(f'ERROR parsing response: {e}')
" 2>&1)

echo ""
echo "[3/3] Assistant reply: ${ASSISTANT_MSG}"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Smoke test complete."
echo "  API endpoint: ${BASE_URL}"
echo "  Model:        ${MODEL_NAME}"
echo "============================================================"
