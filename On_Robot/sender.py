#!/usr/bin/env python3

import collections
import contextlib
import ctypes
import logging
import multiprocessing
import os
import pickle
import select
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List, Set

import numpy as np
import pyrealsense2 as rs


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"none", "off", "disable", "disabled"}:
        return None
    return int(raw)


# Serial numbers of your RealSense cameras.
# Replace these with the actual serial numbers from your devices.
# Find yours with: python3 -c "import pyrealsense2 as rs; [print(d.get_info(rs.camera_info.serial_number)) for d in rs.context().query_devices()]"
CAMERA_SERIALS = (
    "REPLACE_WITH_SERIAL_0",  # front left D455 — change to your camera's serial
    "REPLACE_WITH_SERIAL_1",  # front centre D455 — change to your camera's serial
    "REPLACE_WITH_SERIAL_2",  # front right D455 — change to your camera's serial
    "REPLACE_WITH_SERIAL_3",  # rear D435I — change to your camera's serial
)

# Receiver configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 9999
LATENCY_PROBE_PORT = 10000

# Camera stream configuration
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30
# Keep depth pixels in the same image coordinates used by YOLO on RGB.
ALIGN_DEPTH_TO_COLOR = True
FRAME_TIMEOUT_MS = 2000
CAPTURE_RETRIES = 3
CAPTURE_RETRY_DELAY = 0.05

# Recovery behaviour (DGX Spark / USB re-enumeration friendliness)
STARTUP_HARD_RESET = env_bool("TM_CAMERA_STARTUP_HARD_RESET", True)
STARTUP_RESET_DELAY = float(os.environ.get("TM_CAMERA_STARTUP_RESET_DELAY", "4.0"))
SHUTDOWN_HARD_RESET = env_bool("TM_CAMERA_SHUTDOWN_HARD_RESET", True)
WAIT_FOR_DEVICES_TIMEOUT = 20.0  # seconds

RECOVER_ON_CAPTURE_FAILURE = env_bool("TM_CAMERA_RECOVER_ON_CAPTURE_FAILURE", True)
RECOVERY_RESET_DELAY = 4.0
RECOVERY_RESTART_COOLDOWN = 0.5  # seconds between recovery attempts per camera
MAX_RECOVERIES_PER_CAMERA = 50   # avoid infinite loops on dead hardware

# Network behaviour
SOCKET_SNDBUF = 2**24
CONNECT_TIMEOUT = 5
RECONNECT_INITIAL_BACKOFF = 1.0
RECONNECT_MAX_BACKOFF = 30.0
RECONNECT_AFTER_ERROR_DELAY = 0.5
MESSAGE_HEADER_STRUCT = "<L"  # payload length
MESSAGE_HEADER_SIZE = struct.calcsize(MESSAGE_HEADER_STRUCT)
# Use pickle protocol 5 buffers so large NumPy arrays are sent directly instead
# of copying them into a second multi-megabyte pickle bytes object first.
OUT_OF_BAND_PICKLE_BUFFERS = True
BUFFERED_FRAME_MAGIC = b"TMF5"
BUFFERED_FRAME_HEADER_STRUCT = "<4sIH"  # magic, pickle length, buffer count
BUFFERED_FRAME_BUFFER_LENGTH_STRUCT = "<Q"
# Probe v1 requests carry the monotonic send timestamp plus the previous probe
# RTT.  Probe v2 adds sequence/TAI timestamps for PTP-backed one-way timing
# while preserving v1 compatibility.
LATENCY_PROBE_MAGIC = b"TMLP"
LATENCY_PROBE_VERSION = 2
LATENCY_PROBE_REQUEST_STRUCT = "<QQ"
LATENCY_PROBE_ECHO_STRUCT = "<Q"
LATENCY_PROBE_REQUEST_V2_STRUCT = "<4sHHQQQQ"  # magic, version, flags, seq, mono, tai, previous RTT
LATENCY_PROBE_ECHO_V2_STRUCT = "<4sHHQQQQ"  # magic, version, flags, seq, mono, tai, receiver tai
LATENCY_PROBE_ECHO_SIZE = struct.calcsize(LATENCY_PROBE_ECHO_STRUCT)
LATENCY_PROBE_ECHO_V2_SIZE = struct.calcsize(LATENCY_PROBE_ECHO_V2_STRUCT)
LATENCY_PROBE_RECV_SIZE = 128

# Monitoring
BITRATE_LOG_INTERVAL = 1.0
LATENCY_PROBE_ENABLED = env_bool("TM_PROBE_ENABLED", True)
LATENCY_PROBE_RATE_HZ = 100.0
LATENCY_PROBE_LOG_INTERVAL = float(os.environ.get("TM_PROBE_LOG_INTERVAL", "5.0"))
# Linux CPU isolation for the UDP probe. -1 reserves the highest allowed CPU;
# set to a specific CPU id to choose it, or None to disable affinity.
LATENCY_PROBE_CPU_CORE = env_optional_int("TM_PROBE_CPU_CORE", -1)
RESERVE_LATENCY_PROBE_CPU = env_bool("TM_PROBE_RESERVE_CPU", True)
LATENCY_PROBE_DSCP = int(os.environ.get("TM_PROBE_DSCP", "46"))
LATENCY_PROBE_SOCKET_PRIORITY = int(os.environ.get("TM_PROBE_PRIORITY", "6"))
_probe_bind_device = os.environ.get("TM_PROBE_BIND_DEVICE")
if _probe_bind_device is None:
    _probe_bind_device = os.environ.get("TM_NET_IFACE")
LATENCY_PROBE_BIND_DEVICE = (
    None
    if (_probe_bind_device or "").strip().lower() in {"0", "false", "no", "none", "off", "disable", "disabled"}
    else _probe_bind_device
)
LATENCY_PROBE_RT_PRIORITY = int(os.environ.get("TM_PROBE_RT_PRIORITY", "80"))
LATENCY_PROBE_MLOCK = env_bool("TM_PROBE_MLOCK", True)
LATENCY_PROBE_BUSY_POLL_US = int(os.environ.get("TM_PROBE_BUSY_POLL_US", "50"))
LATENCY_PROBE_BUSY_SPIN_US = int(os.environ.get("TM_PROBE_BUSY_SPIN_US", "200"))
LATENCY_PROBE_SOCKET_BUFFER = int(os.environ.get("TM_PROBE_SOCKET_BUFFER", str(1 << 20)))
CAMERA_METRICS_LOG_INTERVAL = 5.0
PAYLOAD_VERSION = 3

# Python does not expose these on every platform even when Linux supports them.
SO_BUSY_POLL = getattr(socket, "SO_BUSY_POLL", 46)
SO_PREFER_BUSY_POLL = getattr(socket, "SO_PREFER_BUSY_POLL", 69)
SO_INCOMING_CPU = getattr(socket, "SO_INCOMING_CPU", 49)
MCL_CURRENT = 1
MCL_FUTURE = 2


def select_latency_probe_cpu(
    available_cpu_ids: Sequence[int],
    configured_cpu: Optional[int],
) -> Optional[int]:
    """Choose the CPU id reserved for latency probes."""
    if configured_cpu is None:
        return None
    cpu_ids = tuple(sorted({int(cpu_id) for cpu_id in available_cpu_ids}))
    if not cpu_ids:
        return None
    if int(configured_cpu) < 0:
        return cpu_ids[-1]
    if int(configured_cpu) in cpu_ids:
        return int(configured_cpu)
    return None


def reserve_latency_probe_cpu() -> Optional[int]:
    """Keep future app worker threads off the CPU used by the UDP probe."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    try:
        app_cpu_ids = set(os.sched_getaffinity(0))
    except OSError as exc:
        logging.warning("Unable to read sender CPU affinity for latency probe: %s", exc)
        return None

    probe_cpu = select_latency_probe_cpu(app_cpu_ids, LATENCY_PROBE_CPU_CORE)
    if probe_cpu is None:
        if LATENCY_PROBE_CPU_CORE is not None:
            logging.warning(
                "Sender latency probe CPU %s is not in allowed CPU set %s",
                LATENCY_PROBE_CPU_CORE,
                sorted(app_cpu_ids),
            )
        return None
    if not RESERVE_LATENCY_PROBE_CPU or len(app_cpu_ids) < 2:
        return probe_cpu

    try:
        os.sched_setaffinity(0, app_cpu_ids - {probe_cpu})
        logging.info("Sender reserving CPU %d for latency probes", probe_cpu)
    except OSError as exc:
        logging.warning("Unable to reserve sender latency probe CPU %d: %s", probe_cpu, exc)
        return None
    return probe_cpu


def pin_current_thread_to_cpu(cpu_core: Optional[int], name: str) -> None:
    """Pin the calling Linux thread to one CPU when affinity is enabled."""
    if cpu_core is None or not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, {cpu_core})
    except OSError as exc:
        logging.warning("Unable to pin %s to CPU %d: %s", name, cpu_core, exc)


def configure_realtime_thread(priority: int, name: str) -> None:
    """Best-effort SCHED_FIFO for the latency probe thread/process."""
    if priority <= 0 or not hasattr(os, "sched_setscheduler"):
        return
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
        logging.info("%s using SCHED_FIFO priority %d", name, priority)
    except OSError as exc:
        logging.warning("Unable to set %s SCHED_FIFO priority %d: %s", name, priority, exc)


def lock_process_memory(name: str) -> None:
    """Best-effort mlockall to avoid page-fault spikes in the probe loop."""
    if not LATENCY_PROBE_MLOCK:
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.mlockall(MCL_CURRENT | MCL_FUTURE) != 0:
            errno = ctypes.get_errno()
            logging.warning("Unable to mlockall for %s: errno %d", name, errno)
        else:
            logging.info("%s locked current/future memory", name)
    except Exception as exc:
        logging.warning("Unable to mlockall for %s: %s", name, exc)


def clock_tai_ns() -> int:
    if hasattr(time, "CLOCK_TAI"):
        try:
            return time.clock_gettime_ns(time.CLOCK_TAI)
        except OSError:
            return 0
    return 0


def configure_latency_probe_socket(
    sock: socket.socket,
    name: str,
    incoming_cpu: Optional[int] = None,
) -> None:
    """Apply app-level QoS and low-latency socket options for the UDP probe."""
    tos = max(0, min(63, LATENCY_PROBE_DSCP)) << 2
    options = (
        (socket.IPPROTO_IP, socket.IP_TOS, tos, "IP_TOS"),
        (socket.SOL_SOCKET, socket.SO_PRIORITY, LATENCY_PROBE_SOCKET_PRIORITY, "SO_PRIORITY"),
        (socket.SOL_SOCKET, socket.SO_SNDBUF, LATENCY_PROBE_SOCKET_BUFFER, "SO_SNDBUF"),
        (socket.SOL_SOCKET, socket.SO_RCVBUF, LATENCY_PROBE_SOCKET_BUFFER, "SO_RCVBUF"),
    )
    for level, option, value, label in options:
        with contextlib.suppress(OSError):
            sock.setsockopt(level, option, value)
            logging.debug("%s set %s=%s", name, label, value)

    if LATENCY_PROBE_BIND_DEVICE:
        try:
            sock.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_BINDTODEVICE,
                LATENCY_PROBE_BIND_DEVICE.encode() + b"\0",
            )
            logging.info("%s bound to %s", name, LATENCY_PROBE_BIND_DEVICE)
        except OSError as exc:
            logging.warning("%s unable to bind to %s: %s", name, LATENCY_PROBE_BIND_DEVICE, exc)

    if LATENCY_PROBE_BUSY_POLL_US > 0:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, SO_BUSY_POLL, LATENCY_PROBE_BUSY_POLL_US)
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, SO_PREFER_BUSY_POLL, 1)
    if incoming_cpu is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, SO_INCOMING_CPU, int(incoming_cpu))


def query_connected_serials(ctx: rs.context) -> Set[str]:
    serials: Set[str] = set()
    try:
        for dev in ctx.query_devices():
            with contextlib.suppress(Exception):
                serials.add(dev.get_info(rs.camera_info.serial_number))
    except Exception:
        # If the USB stack is in a weird state, query_devices itself can throw.
        return set()
    return serials


def wait_for_serials(serials: Sequence[str], timeout_s: float) -> bool:
    """Wait until all requested serials are visible to librealsense."""
    if not serials:
        return True
    target = set(serials)
    deadline = time.time() + max(timeout_s, 0.0)
    ctx = rs.context()
    while time.time() < deadline:
        seen = query_connected_serials(ctx)
        missing = target - seen
        if not missing:
            return True
        time.sleep(0.2)
    return False


def hardware_reset_all(serials: Sequence[str], reset_delay: float = 4.0) -> None:
    """Best-effort hardware reset for all RealSense devices matching `serials`.

    This is equivalent to a USB unplug/replug at script start and helps
    recover from cases where a previous run crashed or left devices busy.
    """
    try:
        ctx = rs.context()
    except Exception as exc:
        logging.warning("Unable to create RealSense context for hardware reset: %s", exc)
        return

    try:
        dev_list = ctx.query_devices()
    except Exception as exc:
        logging.warning("Unable to query RealSense devices for hardware reset: %s", exc)
        return

    if len(dev_list) == 0:
        logging.warning("No RealSense devices found for hardware reset")
        return

    target_serials = set(serials)
    for dev in dev_list:
        try:
            sn = dev.get_info(rs.camera_info.serial_number)
        except Exception:
            sn = "<unknown>"
        if target_serials and sn not in target_serials:
            continue
        try:
            logging.debug("Sending hardware reset to RealSense device %s", sn)
            dev.hardware_reset()
        except Exception as exc:
            logging.warning("Hardware reset failed for device %s: %s", sn, exc)

    if reset_delay > 0:
        logging.info("Waiting %.1f seconds for RealSense devices to re-enumerate...", reset_delay)
        time.sleep(reset_delay)


class Metrics:
    """Thread-safe container for throughput metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: "collections.deque[Tuple[float, int]]" = collections.deque()
        self._total_bytes_sent = 0

    def add_bytes(self, count: int) -> None:
        now = time.time()
        with self._lock:
            self._samples.append((now, count))
            self._total_bytes_sent += count
            self._expire_locked(now)

    def bitrate_mbps(self) -> float:
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            total_bytes = self._total_bytes_sent
            if not self._samples:
                return 0.0
            earliest = self._samples[0][0]
        elapsed = max(now - earliest, 1e-9)
        return (total_bytes * 8) / (elapsed * 1024 * 1024)

    def _expire_locked(self, now: float) -> None:
        cutoff = now - 3.0
        while self._samples and self._samples[0][0] < cutoff:
            _, count = self._samples.popleft()
            self._total_bytes_sent -= count


class LatencyProbeSender(multiprocessing.Process):
    """Sends tiny UDP echo probes and reports sender-measured RTTs."""

    def __init__(
        self,
        host: str,
        port: int,
        stop_event,
        cpu_core: Optional[int] = None,
    ) -> None:
        super().__init__(name="LatencyProbeSender", daemon=True)
        self._target = (host, port)
        self._stop_event = stop_event
        self._cpu_core = cpu_core
        self._probe_sent_count = 0
        self._probe_echo_count = 0
        self._probe_last_rtt_ns = 0

    def run(self) -> None:
        # The parent sender handles Ctrl+C and signals this child to stop.
        # Ignoring SIGINT here avoids a multiprocessing child traceback when
        # the terminal sends the interrupt to the whole process group.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            pin_current_thread_to_cpu(self._cpu_core, self.name)
            configure_realtime_thread(LATENCY_PROBE_RT_PRIORITY, self.name)
            lock_process_memory(self.name)
            interval_s = 1.0 / LATENCY_PROBE_RATE_HZ
            next_send_s = time.monotonic()
            next_log_s = time.monotonic() + LATENCY_PROBE_LOG_INTERVAL
            pending_round_trip_ns = 0
            sequence = 0
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                configure_latency_probe_socket(sock, self.name, self._cpu_core)
                with contextlib.suppress(OSError):
                    sock.connect(self._target)
                sock.setblocking(False)
                logging.info(
                    "Sending %.0f UDP latency probes/s to %s:%s",
                    LATENCY_PROBE_RATE_HZ,
                    self._target[0],
                    self._target[1],
                )
                while not self._stop_event.is_set():
                    pending_round_trip_ns = self._wait_for_probe_time(
                        sock,
                        next_send_s,
                        pending_round_trip_ns,
                    )
                    if self._stop_event.is_set():
                        break
                    try:
                        sequence += 1
                        send_time_ns = time.monotonic_ns()
                        sock.send(
                            struct.pack(
                                LATENCY_PROBE_REQUEST_V2_STRUCT,
                                LATENCY_PROBE_MAGIC,
                                LATENCY_PROBE_VERSION,
                                0,
                                sequence,
                                send_time_ns,
                                clock_tai_ns(),
                                pending_round_trip_ns,
                            )
                        )
                        self._probe_sent_count += 1
                        pending_round_trip_ns = 0
                    except OSError as exc:
                        logging.warning("Unable to send latency probe: %s", exc)

                    if LATENCY_PROBE_LOG_INTERVAL > 0 and time.monotonic() >= next_log_s:
                        rtt_ms = self._probe_last_rtt_ns / 1_000_000.0
                        logging.info(
                            "Latency probe stats: sent=%d echoes=%d last_rtt_ms=%.3f target=%s:%s",
                            self._probe_sent_count,
                            self._probe_echo_count,
                            rtt_ms,
                            self._target[0],
                            self._target[1],
                        )
                        next_log_s = time.monotonic() + LATENCY_PROBE_LOG_INTERVAL

                    next_send_s += interval_s
                    if next_send_s < time.monotonic():
                        next_send_s = time.monotonic() + interval_s
        except KeyboardInterrupt:
            return

    def _wait_for_probe_time(
        self,
        sock: socket.socket,
        next_send_s: float,
        pending_round_trip_ns: int,
    ) -> int:
        while not self._stop_event.is_set():
            remaining_s = next_send_s - time.monotonic()
            if remaining_s <= 0:
                return self._drain_echoes(sock, pending_round_trip_ns)
            if remaining_s <= (LATENCY_PROBE_BUSY_SPIN_US / 1_000_000.0):
                continue
            try:
                readable, _writable, _error = select.select(
                    [sock],
                    [],
                    [],
                    min(remaining_s, 0.001),
                )
            except OSError:
                return pending_round_trip_ns
            if readable:
                pending_round_trip_ns = self._drain_echoes(
                    sock,
                    pending_round_trip_ns,
                )
        return pending_round_trip_ns

    def _drain_echoes(self, sock: socket.socket, pending_round_trip_ns: int) -> int:
        while True:
            try:
                packet = sock.recv(LATENCY_PROBE_RECV_SIZE)
            except BlockingIOError:
                return pending_round_trip_ns
            except OSError:
                return pending_round_trip_ns
            if len(packet) == LATENCY_PROBE_ECHO_V2_SIZE and packet.startswith(LATENCY_PROBE_MAGIC):
                (
                    _magic,
                    version,
                    _flags,
                    _sequence,
                    send_time_ns,
                    _send_tai_ns,
                    _receiver_tai_ns,
                ) = struct.unpack(LATENCY_PROBE_ECHO_V2_STRUCT, packet)
                if version != LATENCY_PROBE_VERSION:
                    continue
                pending_round_trip_ns = max(time.monotonic_ns() - send_time_ns, 0)
                self._probe_echo_count += 1
                self._probe_last_rtt_ns = pending_round_trip_ns
                continue
            if len(packet) != LATENCY_PROBE_ECHO_SIZE:
                continue
            (send_time_ns,) = struct.unpack(LATENCY_PROBE_ECHO_STRUCT, packet)
            pending_round_trip_ns = max(time.monotonic_ns() - send_time_ns, 0)
            self._probe_echo_count += 1
            self._probe_last_rtt_ns = pending_round_trip_ns


class RealSenseCamera:
    """Encapsulates a single RealSense pipeline with recovery helpers."""

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self._lock = threading.Lock()
        self._pipeline: Optional[rs.pipeline] = None
        self._config = rs.config()
        self._config.enable_device(serial)
        self._config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FRAME_RATE)
        self._config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.rgb8, FRAME_RATE)
        self._depth_to_color_align = (
            rs.align(rs.stream.color) if ALIGN_DEPTH_TO_COLOR else None
        )

        self._last_recovery_ts = 0.0
        self._recovery_count = 0

    def start(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return
            logging.debug("Starting camera %s", self.serial)
            pipe = rs.pipeline()
            pipe.start(self._config)
            self._pipeline = pipe

    def stop(self) -> None:
        with self._lock:
            pipe = self._pipeline
            self._pipeline = None
        if pipe is None:
            return
        logging.debug("Stopping camera %s", self.serial)
        with contextlib.suppress(Exception):
            pipe.stop()
        # Give the driver a moment to settle (helps on some ARM/USB stacks)
        time.sleep(0.05)

    def capture(self) -> Dict[str, Any]:
        with self._lock:
            pipe = self._pipeline
        if pipe is None:
            raise RuntimeError(f"Camera {self.serial} pipeline is not started")

        for attempt in range(1, CAPTURE_RETRIES + 1):
            try:
                frames = pipe.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
                if self._depth_to_color_align is not None:
                    frames = self._depth_to_color_align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if depth_frame and color_frame:
                    return {
                        "capture_received_time_ns": time.time_ns(),
                        "depth_frame_number": int(depth_frame.get_frame_number()),
                        "color_frame_number": int(color_frame.get_frame_number()),
                        "depth_device_timestamp_ms": float(depth_frame.get_timestamp()),
                        "color_device_timestamp_ms": float(color_frame.get_timestamp()),
                        # Sender and capture threads run independently. Own the
                        # bytes before librealsense advances the next frame.
                        "depth": np.asanyarray(depth_frame.get_data()).copy(),
                        "color": np.asanyarray(color_frame.get_data()).copy(),
                    }
                logging.warning(
                    "Camera %s missing depth/color frames (attempt %d/%d)",
                    self.serial, attempt, CAPTURE_RETRIES
                )
            except RuntimeError as exc:
                logging.warning(
                    "Camera %s frame timeout (attempt %d/%d): %s",
                    self.serial, attempt, CAPTURE_RETRIES, exc
                )
            except rs.error as exc:
                logging.warning(
                    "Camera %s RealSense error (attempt %d/%d): %s",
                    self.serial, attempt, CAPTURE_RETRIES, exc
                )
            time.sleep(CAPTURE_RETRY_DELAY)

        raise RuntimeError(f"Failed to capture frames from camera {self.serial}")

    def can_recover_now(self) -> bool:
        if self._recovery_count >= MAX_RECOVERIES_PER_CAMERA:
            return False
        now = time.time()
        return (now - self._last_recovery_ts) >= RECOVERY_RESTART_COOLDOWN

    @property
    def recovery_count(self) -> int:
        return self._recovery_count

    def recover(self) -> None:
        """Stop pipeline, hardware reset the device, wait for re-enumeration, restart."""
        if not self.can_recover_now():
            return
        self._last_recovery_ts = time.time()
        self._recovery_count += 1

        logging.warning("Recovering camera %s (recovery #%d)", self.serial, self._recovery_count)

        # Stop first to release resources cleanly
        self.stop()

        # Hardware reset (best-effort)
        hardware_reset_all([self.serial], reset_delay=RECOVERY_RESET_DELAY)

        # Ensure it is visible again before starting pipeline
        if not wait_for_serials([self.serial], timeout_s=WAIT_FOR_DEVICES_TIMEOUT):
            logging.error("Camera %s did not re-enumerate after reset", self.serial)
            return

        # Restart
        with contextlib.suppress(Exception):
            self.start()


class CameraStreamStats:
    """Interval statistics for one camera capture and send pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._capture_count = 0
        self._send_count = 0
        self._slot_drops = 0
        self._reconnects = 0
        self._capture_wait_samples_s: List[float] = []

    def record_capture(self, wait_s: float) -> None:
        with self._lock:
            self._capture_count += 1
            self._capture_wait_samples_s.append(wait_s)

    def record_send(self) -> None:
        with self._lock:
            self._send_count += 1

    def record_slot_drop(self) -> None:
        with self._lock:
            self._slot_drops += 1

    def record_reconnect(self) -> None:
        with self._lock:
            self._reconnects += 1

    def take_interval(self) -> Tuple[int, int, int, int, List[float]]:
        with self._lock:
            result = (
                self._capture_count,
                self._send_count,
                self._slot_drops,
                self._reconnects,
                self._capture_wait_samples_s,
            )
            self._capture_count = 0
            self._send_count = 0
            self._slot_drops = 0
            self._reconnects = 0
            self._capture_wait_samples_s = []
        return result


class LatestCameraFrameSlot:
    """Single-slot latest-frame handoff between capture and TCP send."""

    def __init__(self, stats: CameraStreamStats) -> None:
        self._stats = stats
        self._condition = threading.Condition()
        self._packet: Optional[Dict[str, Any]] = None

    def put(self, packet: Dict[str, Any]) -> None:
        with self._condition:
            if self._packet is not None:
                self._stats.record_slot_drop()
            self._packet = packet
            self._condition.notify()

    def take(self, stop_event: threading.Event) -> Optional[Dict[str, Any]]:
        with self._condition:
            while self._packet is None and not stop_event.is_set():
                self._condition.wait(timeout=0.2)
            packet = self._packet
            self._packet = None
        return packet

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()


class CameraCaptureWorker(threading.Thread):
    """Captures one free-running RealSense camera into its latest-frame slot."""

    def __init__(
        self,
        camera_id: int,
        camera: RealSenseCamera,
        slot: LatestCameraFrameSlot,
        stats: CameraStreamStats,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"CameraCapture-{camera_id}", daemon=True)
        self._camera_id = camera_id
        self._camera = camera
        self._slot = slot
        self._stats = stats
        self._stop_event = stop_event
        self._sequence = 0

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    self._camera.start()
                    wait_started_s = time.perf_counter()
                    captured = self._camera.capture()
                    self._stats.record_capture(time.perf_counter() - wait_started_s)
                except Exception as exc:
                    logging.warning(
                        "Camera %d (%s) capture failed: %s",
                        self._camera_id,
                        self._camera.serial,
                        exc,
                    )
                    if RECOVER_ON_CAPTURE_FAILURE and self._camera.can_recover_now():
                        self._camera.recover()
                    self._stop_event.wait(RECONNECT_AFTER_ERROR_DELAY)
                    continue

                self._sequence += 1
                self._slot.put(
                    {
                        "version": PAYLOAD_VERSION,
                        "camera_id": self._camera_id,
                        "camera_serial": self._camera.serial,
                        "sequence": self._sequence,
                        **captured,
                    }
                )
        finally:
            self._camera.stop()
            self._slot.wake()


class CameraTcpSender(threading.Thread):
    """Sends latest frames for one camera on its own TCP stream."""

    def __init__(
        self,
        camera_id: int,
        camera_serial: str,
        slot: LatestCameraFrameSlot,
        stats: CameraStreamStats,
        metrics: Metrics,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"CameraSender-{camera_id}", daemon=True)
        self._camera_id = camera_id
        self._camera_serial = camera_serial
        self._slot = slot
        self._stats = stats
        self._metrics = metrics
        self._stop_event = stop_event
        self._sock: Optional[socket.socket] = None

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self._sock is None and not self._connect_with_backoff():
                    break

                packet = self._slot.take(self._stop_event)
                if packet is None:
                    continue

                try:
                    self._send_packet(packet)
                except ConnectionError as exc:
                    logging.warning(
                        "Camera %d (%s) TCP stream lost: %s",
                        self._camera_id,
                        self._camera_serial,
                        exc,
                    )
                    self._stats.record_reconnect()
                    self._close_socket()
                    self._stop_event.wait(RECONNECT_AFTER_ERROR_DELAY)
        finally:
            self._close_socket()

    def _connect_with_backoff(self) -> bool:
        backoff = RECONNECT_INITIAL_BACKOFF
        while not self._stop_event.is_set():
            try:
                sock = socket.create_connection(
                    (SERVER_HOST, SERVER_PORT),
                    timeout=CONNECT_TIMEOUT,
                )
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_SNDBUF)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                with contextlib.suppress(Exception):
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(None)
                self._sock = sock
                logging.debug(
                    "Camera %d (%s) TCP stream connected to %s:%s",
                    self._camera_id,
                    self._camera_serial,
                    SERVER_HOST,
                    SERVER_PORT,
                )
                return True
            except OSError as exc:
                logging.warning(
                    "Camera %d (%s) connection failed: %s. Retrying in %.1f seconds",
                    self._camera_id,
                    self._camera_serial,
                    exc,
                    backoff,
                )
                self._stats.record_reconnect()
                if self._stop_event.wait(backoff):
                    return False
                backoff = min(backoff * 2, RECONNECT_MAX_BACKOFF)
        return False

    def _send_packet(self, packet: Dict[str, Any]) -> None:
        if self._sock is None:
            raise ConnectionError("socket is not connected")
        payload_size, payload_parts = self._build_payload_parts(packet)
        header = struct.pack(MESSAGE_HEADER_STRUCT, payload_size)
        try:
            self._sock.sendall(header)
            for payload_part in payload_parts:
                self._sock.sendall(payload_part)
        except OSError as exc:
            raise ConnectionError(exc) from exc
        self._metrics.add_bytes(MESSAGE_HEADER_SIZE + payload_size)
        self._stats.record_send()

    @staticmethod
    def _build_payload_parts(packet: Dict[str, Any]) -> Tuple[int, Tuple[Any, ...]]:
        if not OUT_OF_BAND_PICKLE_BUFFERS:
            payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
            return len(payload), (payload,)

        pickle_buffers = []
        pickle_payload = pickle.dumps(
            packet,
            protocol=pickle.HIGHEST_PROTOCOL,
            buffer_callback=pickle_buffers.append,
        )
        raw_buffers = tuple(pickle_buffer.raw() for pickle_buffer in pickle_buffers)
        buffered_header = struct.pack(
            BUFFERED_FRAME_HEADER_STRUCT,
            BUFFERED_FRAME_MAGIC,
            len(pickle_payload),
            len(raw_buffers),
        )
        buffer_lengths = b"".join(
            struct.pack(BUFFERED_FRAME_BUFFER_LENGTH_STRUCT, buffer.nbytes)
            for buffer in raw_buffers
        )
        payload_parts = (buffered_header + buffer_lengths, pickle_payload, *raw_buffers)
        payload_size = sum(
            payload_part.nbytes
            if isinstance(payload_part, memoryview)
            else len(payload_part)
            for payload_part in payload_parts
        )
        return payload_size, payload_parts

    def _close_socket(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        with contextlib.suppress(Exception):
            sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(Exception):
            sock.close()


@dataclass
class CameraPipeline:
    camera_id: int
    camera: RealSenseCamera
    stats: CameraStreamStats
    slot: LatestCameraFrameSlot
    capture_worker: CameraCaptureWorker
    sender: CameraTcpSender

    @classmethod
    def build(
        cls,
        camera_id: int,
        camera: RealSenseCamera,
        metrics: Metrics,
        stop_event: threading.Event,
    ) -> "CameraPipeline":
        stats = CameraStreamStats()
        slot = LatestCameraFrameSlot(stats)
        return cls(
            camera_id=camera_id,
            camera=camera,
            stats=stats,
            slot=slot,
            capture_worker=CameraCaptureWorker(
                camera_id, camera, slot, stats, stop_event
            ),
            sender=CameraTcpSender(
                camera_id, camera.serial, slot, stats, metrics, stop_event
            ),
        )

    def start(self) -> None:
        self.capture_worker.start()
        self.sender.start()

    def join(self) -> None:
        self.slot.wake()
        self.capture_worker.join(timeout=FRAME_TIMEOUT_MS / 1000.0 + 2.0)
        self.sender.join(timeout=5.0)


class MultiCameraFrameSender(threading.Thread):
    """Supervises independent per-camera capture and TCP stream pipelines."""

    def __init__(self, cameras: Sequence[RealSenseCamera], metrics: Metrics, stop_event: threading.Event) -> None:
        super().__init__(name="FrameSender", daemon=True)
        self._cameras = list(cameras)
        self._metrics = metrics
        self._stop_event = stop_event
        self._pipelines = [
            CameraPipeline.build(camera_id, camera, metrics, stop_event)
            for camera_id, camera in enumerate(self._cameras)
        ]

    def run(self) -> None:
        if STARTUP_HARD_RESET:
            hardware_reset_all(CAMERA_SERIALS, reset_delay=STARTUP_RESET_DELAY)
        try:
            visible_serials = query_connected_serials(rs.context())
        except Exception as exc:
            logging.warning("Unable to query cameras at startup: %s", exc)
            visible_serials = set()
        missing_serials = set(CAMERA_SERIALS) - visible_serials
        if missing_serials:
            logging.warning(
                "Configured cameras missing at startup; their workers will retry independently: %s",
                sorted(missing_serials),
            )

        for pipeline in self._pipelines:
            pipeline.start()

        try:
            while not self._stop_event.wait(CAMERA_METRICS_LOG_INTERVAL):
                self._clear_pipeline_metrics()
        finally:
            logging.info("Frame sender shutting down")
            self._stop_event.set()
            for pipeline in self._pipelines:
                pipeline.join()
            if SHUTDOWN_HARD_RESET:
                hardware_reset_all(CAMERA_SERIALS, reset_delay=2.0)

    def _clear_pipeline_metrics(self) -> None:
        for pipeline in self._pipelines:
            pipeline.stats.take_interval()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    configure_logging()
    probe_cpu_core = reserve_latency_probe_cpu() if LATENCY_PROBE_ENABLED else None

    # Optional: enable librealsense internal logging (handy on DGX/ARM for debugging)
    # rs.log_to_console(rs.log_severity.warn)

    stop_event = threading.Event()
    probe_stop_event = multiprocessing.Event() if LATENCY_PROBE_ENABLED else None
    metrics = Metrics()
    cameras = [RealSenseCamera(serial) for serial in CAMERA_SERIALS]
    sender = MultiCameraFrameSender(cameras, metrics, stop_event)
    probe_sender: Optional[LatencyProbeSender] = None
    if LATENCY_PROBE_ENABLED and probe_stop_event is not None:
        probe_sender = LatencyProbeSender(
            SERVER_HOST,
            LATENCY_PROBE_PORT,
            probe_stop_event,
            cpu_core=probe_cpu_core,
        )
        # Start the probe process before sender worker threads so it does not fork
        # a live RealSense/threading process.
        probe_sender.start()
    else:
        logging.info("Built-in UDP latency probe disabled")
    sender.start()

    try:
        while sender.is_alive():
            if stop_event.wait(BITRATE_LOG_INTERVAL):
                break
            logging.info("Current bitrate: %.2f Mbps", metrics.bitrate_mbps())
        if not sender.is_alive() and not stop_event.is_set():
            logging.error("Frame sender thread terminated unexpectedly")
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received, stopping sender...")
    finally:
        stop_event.set()
        if probe_stop_event is not None:
            probe_stop_event.set()
        if probe_sender is not None:
            probe_sender.join(timeout=5)
            if probe_sender.is_alive():
                logging.warning("Latency probe sender did not shut down cleanly")
        sender.join(timeout=10)
        if sender.is_alive():
            logging.warning("Frame sender did not shut down cleanly")
        logging.info("Sender shutdown complete")


if __name__ == "__main__":
    main()
