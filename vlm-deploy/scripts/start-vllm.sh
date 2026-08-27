#!/usr/bin/env bash
# =============================================================================
# start-vllm.sh — Start vLLM OpenAI-compatible server for Qwen3.6-35b-a3b
# =============================================================================
# This script is the container ENTRYPOINT. It reads environment variables
# and constructs the vLLM serve command with appropriate flags.
#
# Environment variables (set via .env, compose.yaml, or docker run):
#   MODEL_PATH              — local model path (default: /models/Qwen3.6-35B-A3B)
#   SERVED_MODEL_NAME       — name exposed in /v1/models (default: qwen3.6-35b-a3b)
#   HOST                    — bind address (default: 0.0.0.0)
#   PORT                    — listen port (default: 8000)
#   GPU_MEMORY_UTILIZATION  — GPU memory fraction (default: 0.60)
#   MAX_MODEL_LEN           — max sequence length (default: 4096)
#   MAX_NUM_SEQS            — concurrent scheduler sequences (default: 8)
#   PERFORMANCE_MODE        — runtime tuning profile (default: interactivity)
#   ENABLE_PREFIX_CACHING   — share KV prefix work (default: true)
#   LIMIT_MM_PER_PROMPT     — per-request media cap (default: {"image":1})
#   DEFAULT_CHAT_TEMPLATE_KWARGS — server chat-template defaults
#   TENSOR_PARALLEL_SIZE    — number of GPUs for tensor parallelism (default: 1)
#   ENABLE_MTP              — enable MTP speculative decoding (default: true)
#   MTP_SPECULATIVE_TOKENS  — MTP draft depth (default: 1)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-interactivity}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
if [ -z "${LIMIT_MM_PER_PROMPT:-}" ]; then
    LIMIT_MM_PER_PROMPT='{"image":1}'
fi
if [ -z "${DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]; then
    DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
ENABLE_MTP="${ENABLE_MTP:-true}"
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-1}"

echo "============================================================"
echo "  vLLM Server — Qwen3.6-35b-a3b"
echo "============================================================"
echo "  Model path:            ${MODEL_PATH}"
echo "  Served model name:     ${SERVED_MODEL_NAME}"
echo "  Host:                  ${HOST}"
echo "  Port:                  ${PORT}"
echo "  GPU memory utilization:${GPU_MEMORY_UTILIZATION}"
echo "  Max model length:      ${MAX_MODEL_LEN}"
echo "  Max num seqs:          ${MAX_NUM_SEQS}"
echo "  Performance mode:      ${PERFORMANCE_MODE}"
echo "  Prefix caching:        ${ENABLE_PREFIX_CACHING}"
echo "  Media limit:           ${LIMIT_MM_PER_PROMPT}"
echo "  Chat template kwargs:  ${DEFAULT_CHAT_TEMPLATE_KWARGS}"
echo "  Tensor parallel size:  ${TENSOR_PARALLEL_SIZE}"
echo "  MTP enabled:           ${ENABLE_MTP}"
echo "  MTP speculative tokens:${MTP_SPECULATIVE_TOKENS}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Verify model files exist
# ---------------------------------------------------------------------------
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: Model config not found at ${MODEL_PATH}/config.json"
    echo "       Run scripts/build.sh so the local model is downloaded and baked into the image."
    exit 1
fi

# ---------------------------------------------------------------------------
# Build vLLM command
# ---------------------------------------------------------------------------
CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "${MODEL_PATH}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --host "${HOST}"
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --performance-mode "${PERFORMANCE_MODE}"
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
    --default-chat-template-kwargs "${DEFAULT_CHAT_TEMPLATE_KWARGS}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --trust-remote-code
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 20}'
)

# ---------------------------------------------------------------------------
# Prefix caching (optional)
# ---------------------------------------------------------------------------
if [ "${ENABLE_PREFIX_CACHING}" = "true" ]; then
    CMD+=(--enable-prefix-caching)
elif [ "${ENABLE_PREFIX_CACHING}" = "false" ]; then
    CMD+=(--no-enable-prefix-caching)
else
    echo "WARNING: ENABLE_PREFIX_CACHING is '${ENABLE_PREFIX_CACHING}', expected 'true' or 'false'."
fi

# ---------------------------------------------------------------------------
# MTP speculative decoding (optional)
# ---------------------------------------------------------------------------
if [ "${ENABLE_MTP}" = "true" ]; then
    echo "Enabling MTP speculative decoding..."
    CMD+=(
        --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_SPECULATIVE_TOKENS}}"
    )
elif [ "${ENABLE_MTP}" = "false" ]; then
    echo "MTP disabled."
else
    echo "WARNING: ENABLE_MTP is '${ENABLE_MTP}', expected 'true' or 'false'. Disabling MTP."
fi

# ---------------------------------------------------------------------------
# Launch vLLM
# ---------------------------------------------------------------------------
echo ""
echo "Starting vLLM with command:"
echo "  ${CMD[*]}"
echo ""

exec "${CMD[@]}"
