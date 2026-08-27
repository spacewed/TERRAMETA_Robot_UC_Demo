# On_Receiver - Point Cloud Viewer and AI Workers

`receiver.py` receives RealSense depth/RGB frames over TCP, renders a real-time 3D point cloud with OpenGL, and runs the optional AI overlays. The same viewer and AI worker classes are also reused by `On_Robot/local_ai_viewer.py` for robot-local operation.

## Running Networked Receiver Mode

```bash
cd On_Receiver
python3 receiver.py
```

The receiver listens on `0.0.0.0:9999` by default. Start this before `On_Robot/sender.py` in networked split mode.

Robot-local mode does not run this file directly; use:

```bash
python3 On_Robot/local_ai_viewer.py
```

from the repo root.

## Prerequisites

### System Packages

```bash
sudo apt update && sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-numpy \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libsdl2-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libfreetype-dev \
    libportmidi-dev
```

### Python Dependencies

```bash
pip install -r requirements.txt
```

For robot-local mode, installing `On_Robot/requirements.txt` from the repo root also includes these GUI and AI dependencies.

## Controls

| Key / Action | Description |
|--------------|-------------|
| W/A/S/D | Move forward/left/back/right |
| Q/E | Move down/up |
| Left mouse drag | Look around |
| Scroll wheel | Zoom |
| T | Toggle depth heatmap / RGB point colours |
| H | Toggle human YOLO boxes |
| O | Toggle all YOLO/COCO object boxes |
| P | Toggle pose points and skeletons |
| V | Toggle VLM semantic object boxes |
| L | Toggle streamed VLM scene descriptions |
| Click GUI switches | Toggle view mode, YOLO boxes, pose, VLM boxes, or scene text |
| Escape | Quit |

## Frame Input

Networked mode expects protocol-version-3 packets from up to four cameras:

- Depth: `1280x720 @ 30 fps`, `z16`
- Colour: `1280x720 @ 30 fps`, `rgb8`
- Depth aligned to RGB before transmission

Each camera has its own TCP stream. The receiver uploads a camera texture only when a new frame arrives, keeps the last rendered frame for stale cameras, and does not wait for synchronized four-camera bundles.

Robot-local mode bypasses TCP by using a local frame source with the same `pop_pending_frames()` and `snapshot_latest_frames()` interface as `FrameReceiver`.

## AI Features

### YOLO Segmentation Boxes

Press `H` for person-only boxes or `O` for all YOLO/COCO classes. Both modes use the same resident YOLO segmentation worker; all-object mode disables the person-only toggle because it already includes people.

Key settings in [receiver.py](receiver.py):

```python
DETECTION_MODEL_PATH = "yolo11n-seg.pt"
DETECTION_CONF_THRESHOLD = 0.5
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_IMGSZ = 640
DETECTION_DEVICE = "cuda:0"
DETECTION_HALF_PRECISION = True
DETECTION_BATCH_SIZE = 1
DETECTION_MAX_DET = 20
DETECTION_RETINA_MASKS = False
```

The detector runs on CUDA by default and does not silently fall back to CPU. RGB masks are fitted to aligned depth to draw metric 3D boxes.

### Pose Points and Skeletons

Press `P` to run YOLO pose. Valid 2D keypoints are sampled against depth and back-projected into the shared 3D view.

```python
POSE_MODEL_PATH = "yolo11n-pose.pt"
POSE_CONF_THRESHOLD = 0.45
POSE_DEVICE = "cuda:0"
POSE_HALF_PRECISION = True
POSE_BATCH_SIZE = 1
POSE_MAX_DET = 10
```

### VLM Semantic Object Boxes

Press `V` to query an OpenAI-compatible VLM endpoint for low-rate semantic object localization. The VLM returns 2D RGB boxes; the viewer samples the paired depth frame to draw 3D dimensions.

```python
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.6-35b-a3b")
VLM_REQUEST_RATE_HZ = 2.0
VLM_REQUEST_TIMEOUT_S = 5.0
VLM_MAX_TOKENS = 32
VLM_IMAGE_MAX_SIDE_PX = 640
VLM_MAX_OBJECTS = 1
VLM_REQUEST_WORKERS = 2
```

The VLM object path uses independent camera worker shards with one image per request.

### VLM Scene Text

Press `L` to show streamed natural-language scene descriptions in GUI text panels. Descriptions are not printed to CLI. Each configured camera can have one in-flight streamed request, so slow cameras do not block faster cameras from refreshing.

```python
SCENE_DESCRIPTION_REFRESH_INTERVAL_S = 10.0
SCENE_DESCRIPTION_MAX_TOKENS = 256
SCENE_DESCRIPTION_JPEG_QUALITY = 82
SCENE_DESCRIPTION_IMAGE_MAX_SIDE_PX = 512
SCENE_DESCRIPTION_WORKERS = 4
```

The prompt asks for concise 2-4 sentence descriptions and streams partial text into the overlay in real time.

## VLM Endpoint

VLM features require an OpenAI-compatible vision server. The repo includes [../vlm-deploy](../vlm-deploy), tuned for FP8 weights from `Qwen/Qwen3.6-35B-A3B-FP8`.

If the server is not on the same machine as the viewer, set:

```bash
export VLM_BASE_URL=http://<server-ip>:8000/v1
```

before launching `receiver.py` or `local_ai_viewer.py`.

## Metrics

Networked receiver mode logs:

- average camera FPS
- frame-stream bitrate in Mbps
- UDP probe latency and jitter summaries
- connection status

It appends completed 30-second windows to `receiver_30s_metrics.csv`. The latency columns are half-RTT estimates from UDP echo probes, not synchronized one-way measurements.

Robot-local mode reuses the viewer but logs local capture FPS and active camera IDs instead of network metrics.

## Architecture

```text
Networked:
On_Robot/sender.py -> FrameReceiver -> PointCloudApp

Robot-local:
On_Robot/local_ai_viewer.py -> LocalFrameSource -> PointCloudApp

PointCloudApp:
  pygame events and control panel
  moderngl point-cloud renderer
  DetectionWorker for YOLO boxes
  PoseDetectionWorker for pose points
  VLMDetectionWorker for semantic boxes
  SceneDescriptionWorker for streamed scene text
  BBoxRenderer for boxes and skeletons
```

## Testing

From the repo root:

```bash
python3 -m unittest On_Receiver.test_receiver_streams
```
