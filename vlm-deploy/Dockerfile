# =============================================================================
# Dockerfile — vLLM OpenAI-compatible server for Qwen/Qwen3.6-35B-A3B-FP8
# =============================================================================
# Base image: nvcr.io/nvidia/vllm:26.04-py3 (includes vLLM, CUDA, PyTorch)
#
# The model is baked into this image from ./models/Qwen3.6-35B-A3B-FP8.
# Run scripts/build.sh to download it into that ignored local directory first.
# =============================================================================

FROM nvcr.io/nvidia/vllm:26.04-py3

LABEL maintainer="terrameta"
LABEL description="vLLM server for Qwen3.6-35b-a3b with MTP speculative decoding"

# Copy model files before scripts so script-only rebuilds can reuse this large layer.
ARG LOCAL_MODEL_DIR_NAME="Qwen3.6-35B-A3B-FP8"
COPY models/${LOCAL_MODEL_DIR_NAME}/ /models/Qwen3.6-35B-A3B/
RUN test -f /models/Qwen3.6-35B-A3B/config.json

# Copy startup script
COPY scripts/start-vllm.sh /usr/local/bin/start-vllm.sh
RUN chmod +x /usr/local/bin/start-vllm.sh

# Runtime defaults
ENV MODEL_PATH="/models/Qwen3.6-35B-A3B"
ENV SERVED_MODEL_NAME="qwen3.6-35b-a3b"
ENV HOST="0.0.0.0"
ENV PORT="8000"
ENV GPU_MEMORY_UTILIZATION="0.40"
ENV MAX_MODEL_LEN="8192"
ENV TENSOR_PARALLEL_SIZE="1"
ENV ENABLE_MTP="true"

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/start-vllm.sh"]
