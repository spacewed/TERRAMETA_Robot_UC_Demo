#!/usr/bin/env bash
# =============================================================================
# build.sh — Download the model once locally, then bake it into the image
# =============================================================================
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   ./scripts/build.sh
#   # If Docker needs sudo:
#   DOCKER_COMMAND="sudo docker" ./scripts/build.sh
#
# The model is stored under ./models/Qwen3.6-35B-A3B-FP8, which is ignored by Git
# but included in the Docker build context so the Dockerfile can COPY it.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${IMAGE_NAME:-vlm-qwen36-35b-a3b}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3.6-35B-A3B-FP8}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-Qwen3.6-35B-A3B-FP8}"
LOCAL_MODEL_DIR="${PROJECT_DIR}/models/${MODEL_DIR_NAME}"
DOCKER_COMMAND="${DOCKER_COMMAND:-docker}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_VENV_DIR="${HF_VENV_DIR:-${PROJECT_DIR}/.hf-venv}"
read -r -a DOCKER_CMD <<< "${DOCKER_COMMAND}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
USE_SECRET=false
FORCE_DOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --secret) USE_SECRET=true ;;
        --force-download) FORCE_DOWNLOAD=true ;;
        --help|-h)
            echo "Usage: $0 [--secret] [--force-download]"
            echo ""
            echo "Options:"
            echo "  --secret          Accepted for compatibility; token is only used by the local downloader"
            echo "  --force-download  Re-run the Hugging Face download even if model files already exist"
            echo ""
            echo "Environment:"
            echo "  HF_TOKEN    Hugging Face token for model download, if required"
            echo "  MODEL_REPO  Hugging Face model repo (default: ${MODEL_REPO})"
            echo "  MODEL_DIR_NAME  Local model directory under ./models (default: ${MODEL_DIR_NAME})"
            echo "  IMAGE_NAME  Docker image name (default: ${IMAGE_NAME})"
            echo "  IMAGE_TAG   Docker image tag (default: ${IMAGE_TAG})"
            echo "  DOCKER_COMMAND  Docker command (default: docker; e.g. \"sudo docker\")"
            echo "  PYTHON_BIN  Python used for the local Hugging Face venv (default: python3)"
            echo "  HF_VENV_DIR Local downloader venv (default: ${HF_VENV_DIR})"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg"
            exit 1
            ;;
    esac
done

if [ "${EUID}" -eq 0 ] && [ -n "${SUDO_USER:-}" ] && [ "${ALLOW_ROOT_BUILD:-}" != "1" ]; then
    echo "ERROR: Do not run this script with sudo."
    echo ""
    echo "The model download should run as your normal user so files in vlm-deploy/models"
    echo "do not become root-owned. If Docker needs sudo, run:"
    echo ""
    echo "  DOCKER_COMMAND=\"sudo docker\" ./scripts/build.sh"
    echo ""
    echo "If a previous sudo run created root-owned artifacts, repair them with:"
    echo ""
    echo "  sudo chown -R \"\$USER:\$USER\" models .hf-venv 2>/dev/null || true"
    exit 1
fi

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
ensure_writable_model_dir() {
    if [ -e "${PROJECT_DIR}/models" ] && [ ! -w "${PROJECT_DIR}/models" ]; then
        echo "ERROR: ${PROJECT_DIR}/models is not writable by $(id -un)."
        echo "       Repair ownership with:"
        echo "       sudo chown -R \"\$USER:\$USER\" \"${PROJECT_DIR}/models\""
        exit 1
    fi

    mkdir -p "${LOCAL_MODEL_DIR}"

    if [ ! -w "${LOCAL_MODEL_DIR}" ]; then
        echo "ERROR: ${LOCAL_MODEL_DIR} is not writable by $(id -un)."
        echo "       Repair ownership with:"
        echo "       sudo chown -R \"\$USER:\$USER\" \"${PROJECT_DIR}/models\""
        exit 1
    fi

    if ! mkdir -p "${LOCAL_MODEL_DIR}/.cache/huggingface"; then
        echo "ERROR: Could not create the Hugging Face metadata cache under:"
        echo "       ${LOCAL_MODEL_DIR}/.cache/huggingface"
        echo "       Repair ownership with:"
        echo "       sudo chown -R \"\$USER:\$USER\" \"${PROJECT_DIR}/models\""
        exit 1
    fi

    if [ ! -w "${LOCAL_MODEL_DIR}/.cache" ] || [ ! -w "${LOCAL_MODEL_DIR}/.cache/huggingface" ]; then
        echo "ERROR: ${LOCAL_MODEL_DIR}/.cache is not writable by $(id -un)."
        echo "       Repair ownership with:"
        echo "       sudo chown -R \"\$USER:\$USER\" \"${PROJECT_DIR}/models\""
        exit 1
    fi
}

has_model_weights() {
    local weight_file
    weight_file="$(
        find "${LOCAL_MODEL_DIR}" -maxdepth 1 -type f \
            \( -name "*.safetensors" -o -name "*.bin" \) \
            -print -quit 2>/dev/null || true
    )"
    [ -n "$weight_file" ]
}

model_ready() {
    [ -f "${LOCAL_MODEL_DIR}/config.json" ] && has_model_weights
}

download_with_hf_cli() {
    local cli_bin="$1"
    local args=(download "${MODEL_REPO}" --local-dir "${LOCAL_MODEL_DIR}")

    if [ -n "${HF_TOKEN:-}" ]; then
        args+=(--token "${HF_TOKEN}")
    fi

    "$cli_bin" "${args[@]}"
}

download_with_python_bin() {
    local python_bin="$1"

    HF_TOKEN="${HF_TOKEN:-}" \
    MODEL_REPO="${MODEL_REPO}" \
    LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR}" \
    "$python_bin" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_REPO"],
    local_dir=os.environ["LOCAL_MODEL_DIR"],
    token=os.environ.get("HF_TOKEN") or None,
)
PY
}

ensure_hf_venv() {
    if [ ! -x "${HF_VENV_DIR}/bin/python" ]; then
        echo "Creating local Hugging Face downloader venv:"
        echo "  ${HF_VENV_DIR}"
        "${PYTHON_BIN}" -m venv "${HF_VENV_DIR}" || {
            echo "ERROR: Could not create ${HF_VENV_DIR}."
            echo "       Install venv support, for example: sudo apt install python3-venv"
            exit 1
        }
    fi

    if ! "${HF_VENV_DIR}/bin/python" -c "import huggingface_hub" >/dev/null 2>&1; then
        "${HF_VENV_DIR}/bin/python" -m pip install -U pip huggingface_hub
    fi
}

download_model() {
    local downloaded=false

    ensure_writable_model_dir

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "WARNING: HF_TOKEN is not set. Downloading without authentication."
        echo "         If the model requires auth, set: export HF_TOKEN=hf_xxxxx"
    fi

    echo ""
    echo "Downloading ${MODEL_REPO} into:"
    echo "  ${LOCAL_MODEL_DIR}"
    echo ""

    ensure_hf_venv
    if download_with_python_bin "${HF_VENV_DIR}/bin/python"; then
        downloaded=true
    else
        echo "WARNING: local Hugging Face venv download failed; trying fallback downloaders."
    fi

    if [ "$downloaded" = false ] && command -v hf >/dev/null 2>&1; then
        if download_with_hf_cli hf; then
            downloaded=true
        else
            echo "WARNING: hf download failed; trying the next available downloader."
        fi
    fi

    if [ "$downloaded" = false ] && command -v huggingface-cli >/dev/null 2>&1; then
        if download_with_hf_cli huggingface-cli; then
            downloaded=true
        else
            echo "WARNING: huggingface-cli download failed; trying the next available downloader."
        fi
    fi

    if [ "$downloaded" = false ] && "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
        if download_with_python_bin "${PYTHON_BIN}"; then
            downloaded=true
        fi
    fi

    if [ "$downloaded" = false ]; then
        echo "ERROR: Hugging Face model download failed."
        echo "       Check your network access, MODEL_REPO, and HF_TOKEN."
        exit 1
    fi

    if ! model_ready; then
        echo "ERROR: Download completed, but ${LOCAL_MODEL_DIR} is missing config.json or weight shards."
        exit 1
    fi
}

echo "============================================================"
echo "  Building vLLM image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Model repo: ${MODEL_REPO}"
echo "  Local model dir: ${LOCAL_MODEL_DIR}"
echo "  Docker command: ${DOCKER_COMMAND}"
echo "  HF venv dir: ${HF_VENV_DIR}"
echo "  HF_TOKEN: ${HF_TOKEN:+set (hidden)}"
echo "============================================================"

if [ "$USE_SECRET" = true ]; then
    echo "Note: --secret is no longer needed; HF_TOKEN is only used for the local download."
fi

if [ "$FORCE_DOWNLOAD" = true ] || ! model_ready; then
    download_model
else
    echo "Local model files already exist; skipping download."
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
"${DOCKER_CMD[@]}" build \
    --build-arg LOCAL_MODEL_DIR_NAME="${MODEL_DIR_NAME}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${PROJECT_DIR}/Dockerfile" \
    "${PROJECT_DIR}"

echo ""
echo "Build complete: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To run:"
echo "  docker compose up -d"
echo "  # or"
echo "  ./scripts/run.sh"
