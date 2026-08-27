# On_Robot - RealSense Capture Entry Points

This folder contains the robot-side RealSense code. It supports both the original networked sender and the newer robot-local viewer.

- [sender.py](sender.py): captures four RealSense cameras and streams one TCP frame feed per camera to `On_Receiver/receiver.py`.
- [local_ai_viewer.py](local_ai_viewer.py): captures the same cameras locally, then reuses the receiver viewer and AI workers in-process. It does not transmit frames and does not need a receiver process.

## Camera Configuration

Camera serial numbers live in [sender.py](sender.py):

```python
CAMERA_SERIALS = (
    "239622301095",
    "241122303877",
    "241122305526",
    "215322072378",
)
```

Update these for the robot before running either entry point. Find connected cameras with:

```bash
python3 -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]"
```

## Stream Configuration

- Depth: `1280x720 @ 30 fps`, `z16`
- Colour: `1280x720 @ 30 fps`, `rgb8`
- Alignment: depth is aligned to colour by default so RGB detections sample matching depth pixels
- Recovery: capture workers can reset and restart individual cameras after capture failures

## Prerequisites

### System Packages

For RealSense capture on Ubuntu 24.04:

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    libusb-1.0-0-dev \
    libudev-dev \
    python3-dev \
    python3-pip \
    python3-numpy
```

For `local_ai_viewer.py`, also install the GUI/OpenGL packages listed in [../On_Receiver/README.md](../On_Receiver/README.md).

### Udev Rules

If librealsense did not install rules automatically:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="8086", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### librealsense2

This project has been used with librealsense `v2.57.7` built from source on ARM64.

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.57.7
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=false \
         -DBUILD_GRAPHICAL_EXAMPLES=false \
         -DBUILD_WITH_CUDA=false \
         -DBUILD_PYTHON_BINDINGS=true
make -j$(nproc)
sudo make install
sudo ldconfig
```

`pyrealsense2` may be installed by the librealsense build rather than pip. If using a virtual environment, you may need to expose `/usr/local/lib/python3.12/dist-packages` via `PYTHONPATH`.

## Python Dependencies

For either robot entry point:

```bash
pip install -r On_Robot/requirements.txt
```

`On_Robot/requirements.txt` includes [../On_Receiver/requirements.txt](../On_Receiver/requirements.txt), because `local_ai_viewer.py` reuses the receiver GUI and AI stack in-process. Run the command from the repo root, or use `pip install -r requirements.txt` if you are already inside `On_Robot`.

## Running Robot-Local Mode

From the repo root:

```bash
python3 On_Robot/local_ai_viewer.py
```

This opens the point-cloud GUI directly on the robot. It uses:

- `sender.py` RealSense capture and recovery helpers
- `receiver.py` rendering and AI workers
- the same GUI controls as the networked receiver

No TCP frame streams, receiver process, or UDP latency probe are created.

## Running Networked Sender Mode

Start `On_Receiver/receiver.py` on the receiver machine first, then run:

```bash
cd On_Robot
python3 sender.py
```

The sender connects to:

```python
SERVER_HOST = "192.168.3.3"
SERVER_PORT = 9999
LATENCY_PROBE_PORT = 10000
```

Edit these constants in [sender.py](sender.py) for your receiver IP and ports.

## Behaviour

- One capture worker per camera.
- In networked mode, one TCP sender per camera.
- Single-slot latest-frame handoff, so stale frames are dropped when downstream work cannot keep up.
- Startup hardware reset is enabled by default.
- Individual camera recovery can stop, reset, wait for re-enumeration, and restart a failed camera.
- Networked mode runs a separate UDP latency probe process.
- Robot-local mode logs local capture FPS instead of bitrate or network latency.
