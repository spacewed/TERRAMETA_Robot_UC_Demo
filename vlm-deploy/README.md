# vLLM Deployment - Qwen3.6-35B-A3B-FP8

This folder runs the FP8 weights from `Qwen/Qwen3.6-35B-A3B-FP8` as an OpenAI-compatible vLLM server for the viewer's VLM object-box and scene-description features.

The model is downloaded once into an ignored local folder under `vlm-deploy/models/`, then baked into the Docker image. Runtime does not need Hugging Face access or a host model mount.

## Files

```text
vlm-deploy/
  Dockerfile
  compose.yaml
  .env.example
  models/              # ignored; populated by scripts/build.sh
  scripts/
    build.sh
    run.sh
    start-vllm.sh
    smoke-test.sh
  README.md
```

## Prerequisites

- NVIDIA GPU and driver
- Docker with NVIDIA Container Toolkit
- Access to `nvcr.io/nvidia/vllm:26.04-py3`
- Hugging Face CLI tooling for the one-time local download:
  `python3 -m pip install -U huggingface_hub`
- A Hugging Face token if the model requires authentication

The defaults are tuned for a 128 GB DGX Spark-style setup while leaving headroom for YOLO and the viewer:

```text
GPU_MEMORY_UTILIZATION=0.60
MAX_MODEL_LEN=4096
MAX_NUM_SEQS=8
TENSOR_PARALLEL_SIZE=1
```

## Model Location

The build script stores the model here:

```text
vlm-deploy/models/Qwen3.6-35B-A3B-FP8
```

That directory is ignored by Git, but it is intentionally included in the Docker build context. The Dockerfile copies it to:

```text
/models/Qwen3.6-35B-A3B
```

The container checks for `${MODEL_PATH}/config.json` at startup.

## Quick Start

```bash
cd vlm-deploy
cp .env.example .env
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # only if required
./scripts/build.sh
docker compose up -d
```

If Docker requires sudo on the target system, keep the model download as your user and only run the Docker build through sudo:

```bash
DOCKER_COMMAND="sudo docker" ./scripts/build.sh
sudo docker compose up -d
```

Do not run `sudo ./scripts/build.sh`; that makes the downloader use root's Python and can leave `models/` owned by root. If that already happened, fix local ownership first:

```bash
sudo chown -R "$USER:$USER" models .hf-venv 2>/dev/null || true
```

If the shell prompt shows another repo venv, such as `On_Robot/venv`, either deactivate it or let `scripts/build.sh` use its local `.hf-venv`; the deploy script does not need the robot/viewer Python environment.

The server listens on:

```text
http://localhost:8000/v1
```

Check it with:

```bash
./scripts/smoke-test.sh
```

Stop it with:

```bash
./scripts/run.sh --stop
```

## Image Build

`scripts/build.sh` skips the download when `models/Qwen3.6-35B-A3B-FP8/config.json` and weight shards already exist.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
./scripts/build.sh
```

To refresh the local model copy:

```bash
./scripts/build.sh --force-download
```

The image is tagged as:

```text
vlm-qwen36-35b-a3b:latest
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_NAME` | `vlm-qwen36-35b-a3b` | Docker image name |
| `IMAGE_TAG` | `latest` | Docker image tag |
| `MODEL_REPO` | `Qwen/Qwen3.6-35B-A3B-FP8` | Hugging Face repo used by `scripts/build.sh` |
| `MODEL_DIR_NAME` | `Qwen3.6-35B-A3B-FP8` | Local directory under `vlm-deploy/models/` and Docker build source |
| `DOCKER_COMMAND` | `docker` | Docker command used by `scripts/build.sh`; use `sudo docker` when needed |
| `HF_VENV_DIR` | `vlm-deploy/.hf-venv` | Local downloader venv used when Hugging Face tooling is not installed |
| `PYTHON_BIN` | `python3` | Python used to create the downloader venv |
| `MODEL_PATH` | `/models/Qwen3.6-35B-A3B` | Container model path |
| `SERVED_MODEL_NAME` | `qwen3.6-35b-a3b` | Name used by viewer requests |
| `HOST` | `0.0.0.0` | vLLM bind address |
| `PORT` | `8000` | vLLM port |
| `GPU_MEMORY_UTILIZATION` | `0.60` | Fraction of GPU memory reserved by vLLM |
| `MAX_MODEL_LEN` | `4096` | Context length; trimmed because the app does not need a huge context window |
| `MAX_NUM_SEQS` | `8` | Concurrent scheduler sequences; enough for four scene streams plus object requests |
| `PERFORMANCE_MODE` | `interactivity` | Low-latency serving profile |
| `ENABLE_PREFIX_CACHING` | `true` | Reuse repeated prompt prefixes |
| `LIMIT_MM_PER_PROMPT` | `{"image":1}` | One image per VLM request |
| `DEFAULT_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking":false}` | Disable thinking mode by default |
| `TENSOR_PARALLEL_SIZE` | `1` | GPUs used for tensor parallelism |
| `ENABLE_MTP` | `true` | Enable MTP speculative decoding |
| `MTP_SPECULATIVE_TOKENS` | `1` | MTP draft token count |

## Viewer Integration

Both `On_Receiver/receiver.py` and `On_Robot/local_ai_viewer.py` use these defaults:

```python
VLM_BASE_URL = "http://127.0.0.1:8000/v1"
VLM_MODEL = "qwen3.6-35b-a3b"
```

If vLLM runs on another machine:

```bash
export VLM_BASE_URL=http://<server-ip>:8000/v1
```

## API Smoke Tests

List models:

```bash
curl http://localhost:8000/v1/models
```

Text completion:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Describe a safe robot work area in one sentence."}],
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

Image chat requests should use OpenAI chat-completions image messages, matching the receiver code.

## Troubleshooting

### Server will not start

1. Check GPU availability with `nvidia-smi`.
2. Check logs with `docker compose logs -f`.
3. Rebuild with `./scripts/build.sh` so the local model is copied into the image.
4. Set `ENABLE_MTP=false` if speculative decoding causes startup errors.
5. Lower `MAX_MODEL_LEN` if KV cache allocation is too large.

### CUDA out of memory

- Keep `GPU_MEMORY_UTILIZATION=0.60` on 128 GB systems when YOLO also runs on the GPU.
- Reduce `MAX_MODEL_LEN` if the app does not need long context.
- Reduce `MAX_NUM_SEQS` only if concurrency is not needed; keep at least `4` for four parallel scene streams.
- Stop unrelated GPU processes.
- Use tensor parallelism if multiple GPUs are available.

### Slow scene descriptions

- Keep `MAX_NUM_SEQS` at `8` so the four scene streams can be scheduled together.
- Keep `DEFAULT_CHAT_TEMPLATE_KWARGS={"enable_thinking":false}`.
- Keep scene images modest in the viewer (`SCENE_DESCRIPTION_IMAGE_MAX_SIDE_PX=512` by default).
- Use `PERFORMANCE_MODE=interactivity`.

## License

This project is licensed under the [MIT License](../LICENSE).
