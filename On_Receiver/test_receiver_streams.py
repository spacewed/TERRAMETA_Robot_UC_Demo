import pickle
import socket
import struct
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import receiver
from depth_to_3d import _fit_axis_aligned_extents, fit_bbox_only
from pose_to_3d import backproject_keypoints, fit_pose_bbox
from yolo_detector import YoloSegDetector
from vlm_detector import VLMDetectorError, parse_vlm_detections, parse_vlm_line_detections
from vlm_scene_describer import OpenAISceneDescriber, SceneDescription, SceneDescriptionWorker


class ReceiverStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.height_patch = mock.patch.object(receiver, "FRAME_HEIGHT", 3)
        self.width_patch = mock.patch.object(receiver, "FRAME_WIDTH", 4)
        self.height_patch.start()
        self.width_patch.start()

    def tearDown(self) -> None:
        self.width_patch.stop()
        self.height_patch.stop()

    def packet(self, camera_id: int, sequence: int = 1, version: int = 3) -> dict:
        return {
            "version": version,
            "camera_id": camera_id,
            "camera_serial": f"serial-{camera_id}",
            "sequence": sequence,
            "capture_received_time_ns": sequence,
            "depth_frame_number": sequence,
            "color_frame_number": sequence,
            "depth_device_timestamp_ms": float(sequence),
            "color_device_timestamp_ms": float(sequence),
            "depth": np.zeros((receiver.FRAME_HEIGHT, receiver.FRAME_WIDTH), dtype=np.uint16),
            "color": np.zeros(
                (receiver.FRAME_HEIGHT, receiver.FRAME_WIDTH, 3),
                dtype=np.uint8,
            ),
        }

    def frame(self, camera_id: int, sequence: int = 1) -> receiver.CameraFrame:
        return receiver.FrameReceiver._extract_camera_packet(
            self.packet(camera_id, sequence)
        )

    def test_protocol_v3_validation_rejects_bundle_and_bad_arrays(self) -> None:
        frame = self.frame(0)
        self.assertEqual(frame.camera_id, 0)
        self.assertEqual(frame.sequence, 1)

        with self.assertRaisesRegex(ValueError, "unsupported payload version"):
            receiver.FrameReceiver._extract_camera_packet(self.packet(0, version=2))

        invalid = self.packet(0)
        invalid["depth"] = invalid["depth"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "invalid depth array"):
            receiver.FrameReceiver._extract_camera_packet(invalid)

        with self.assertRaisesRegex(ValueError, "expected dict payload"):
            receiver.FrameReceiver._extract_camera_packet(
                [self.packet(0)["depth"], self.packet(0)["color"]]
            )

    def test_buffered_payload_deserializes_packet_arrays(self) -> None:
        packet = self.packet(0)
        pickle_buffers = []
        pickle_payload = pickle.dumps(
            packet,
            protocol=pickle.HIGHEST_PROTOCOL,
            buffer_callback=pickle_buffers.append,
        )
        raw_buffers = [pickle_buffer.raw() for pickle_buffer in pickle_buffers]
        buffered_prefix = struct.pack(
            receiver.BUFFERED_FRAME_HEADER_STRUCT,
            receiver.BUFFERED_FRAME_MAGIC,
            len(pickle_payload),
            len(raw_buffers),
        )
        buffer_lengths = b"".join(
            struct.pack(receiver.BUFFERED_FRAME_BUFFER_LENGTH_STRUCT, buffer.nbytes)
            for buffer in raw_buffers
        )
        payload = b"".join((buffered_prefix, buffer_lengths, pickle_payload, *raw_buffers))

        raw = receiver.FrameReceiver._deserialize_camera_payload(payload)
        frame = receiver.FrameReceiver._extract_camera_packet(raw)

        self.assertEqual(frame.camera_id, 0)
        np.testing.assert_array_equal(frame.depth, packet["depth"])
        np.testing.assert_array_equal(frame.color, packet["color"])

    def test_latest_state_isolated_per_camera_and_counts_overwrites(self) -> None:
        stop_event = threading.Event()
        frame_receiver = receiver.FrameReceiver("", 0, receiver.Metrics(), stop_event)

        frame_receiver._store_camera_frame(self.frame(0, 1), 101)
        frame_receiver._store_camera_frame(self.frame(1, 1), 102)
        pending = frame_receiver.pop_pending_frames()
        self.assertEqual(set(pending), {0, 1})

        frame_receiver._store_camera_frame(self.frame(0, 2), 103)
        frame_receiver._store_camera_frame(self.frame(0, 3), 104)
        pending = frame_receiver.pop_pending_frames()
        latest = frame_receiver.snapshot_latest_frames()

        self.assertEqual(pending[0].sequence, 3)
        self.assertEqual(latest[0].sequence, 3)
        self.assertEqual(latest[1].sequence, 1)
        self.assertEqual(frame_receiver._camera_states[0].replaced_latest, 1)
        self.assertEqual(frame_receiver._camera_states[1].replaced_latest, 0)

    def test_average_camera_fps_uses_received_camera_frames(self) -> None:
        stop_event = threading.Event()
        frame_receiver = receiver.FrameReceiver("", 0, receiver.Metrics(), stop_event)
        frame_receiver._camera_fps_last_sample_s = 100.0

        for sequence in range(1, 4):
            frame_receiver._store_camera_frame(self.frame(0, sequence), 100)
        frame_receiver._store_camera_frame(self.frame(1, 1), 100)

        self.assertAlmostEqual(frame_receiver.take_average_camera_fps(101.0), 2.0)

    def test_probe_latency_estimate_uses_half_round_trip_time(self) -> None:
        self.assertAlmostEqual(
            receiver.round_trip_latency_estimate_s(2_000_000),
            0.001,
        )
        self.assertIsNone(receiver.round_trip_latency_estimate_s(0))
        self.assertAlmostEqual(
            receiver.one_way_latency_estimate_s(1_000_000, 1_250_000),
            0.00025,
        )
        self.assertIsNone(receiver.one_way_latency_estimate_s(1_250_000, 1_000_000))
        self.assertAlmostEqual(
            receiver.bounded_one_way_latency_estimate_s(1_000_000, 1_250_000, 1_000_000),
            0.00025,
        )
        self.assertIsNone(
            receiver.bounded_one_way_latency_estimate_s(1_000_000, 3_000_000, 500_000)
        )

    def test_latency_probe_cpu_selection_prefers_highest_allowed_cpu(self) -> None:
        self.assertEqual(
            receiver.select_latency_probe_cpu((2, 4, 6), -1),
            6,
        )
        self.assertEqual(
            receiver.select_latency_probe_cpu((2, 4, 6), 4),
            4,
        )
        self.assertIsNone(receiver.select_latency_probe_cpu((2, 4, 6), 3))
        self.assertIsNone(receiver.select_latency_probe_cpu((2, 4, 6), None))

    def test_latency_summary_includes_average_and_tail_metrics(self) -> None:
        metrics = receiver.LatencyJitterMetrics()
        metrics.add_sample(
            round_trip_time_ns=2_000_000,
            sender_tai_ns=1_000_000,
            receiver_tai_ns=1_250_000,
        )
        metrics.add_sample(
            round_trip_time_ns=4_000_000,
            sender_tai_ns=2_000_000,
            receiver_tai_ns=2_300_000,
        )

        summary = metrics.summary()
        self.assertIsNotNone(summary["latency_mean_ms"])
        self.assertIsNotNone(summary["latency_p99_ms"])
        self.assertIsNotNone(summary["jitter_mean_ms"])
        self.assertIsNotNone(summary["jitter_p95_ms"])
        self.assertIsNotNone(summary["jitter_p99_ms"])
        self.assertEqual(summary["one_way_count"], 2)
        self.assertIsNotNone(summary["one_way_latency_mean_ms"])
        self.assertIsNotNone(summary["one_way_jitter_mean_ms"])

    def test_movement_keydown_does_not_crash_event_processing(self) -> None:
        app = object.__new__(receiver.PointCloudApp)
        event = receiver.pygame.event.Event(
            receiver.pygame.KEYDOWN,
            key=receiver.K_w,
        )

        with mock.patch.object(receiver.pygame.event, "get", return_value=[event]):
            app._process_events()

    def test_render_frame_draws_yolo_boxes_and_pose_skeletons(self) -> None:
        app = object.__new__(receiver.PointCloudApp)
        app._ctx = mock.Mock()
        app._vao = None
        app._bbox_renderer = mock.Mock()
        app._detection_enabled = True
        app._object_detection_enabled = False
        app._detection_worker = mock.Mock(get_detections=mock.Mock(return_value=[self.box(0)]))
        app._vlm_enabled = False
        app._vlm_workers = []
        app._pose_enabled = True
        pose = mock.Mock()
        app._pose_worker = mock.Mock(get_poses=mock.Mock(return_value=[pose]))
        app._compute_view_matrix = lambda: np.eye(4, dtype=np.float32)
        app._compute_projection_matrix = lambda: np.eye(4, dtype=np.float32)
        app._render_control_panel = lambda: None
        app._render_scene_description = lambda: None

        with mock.patch.object(receiver.pygame.display, "flip"):
            app._render_frame()

        app._bbox_renderer.update_boxes.assert_called_once()
        app._bbox_renderer.render.assert_called_once()
        app._bbox_renderer.update_skeletons.assert_called_once_with([pose])
        app._bbox_renderer.render_skeletons.assert_called_once()

    def test_render_frame_clears_pose_skeletons_when_pose_worker_has_no_poses(self) -> None:
        app = object.__new__(receiver.PointCloudApp)
        app._ctx = mock.Mock()
        app._vao = None
        app._bbox_renderer = mock.Mock()
        app._detection_enabled = False
        app._object_detection_enabled = False
        app._detection_worker = None
        app._vlm_enabled = False
        app._vlm_workers = []
        app._pose_enabled = True
        app._pose_worker = mock.Mock(get_poses=mock.Mock(return_value=[]))
        app._compute_view_matrix = lambda: np.eye(4, dtype=np.float32)
        app._compute_projection_matrix = lambda: np.eye(4, dtype=np.float32)
        app._render_control_panel = lambda: None
        app._render_scene_description = lambda: None

        with mock.patch.object(receiver.pygame.display, "flip"):
            app._render_frame()

        app._bbox_renderer.update_boxes.assert_called_once_with([])
        app._bbox_renderer.update_skeletons.assert_called_once_with([])
        app._bbox_renderer.render_skeletons.assert_not_called()

    def test_control_panel_clicks_dispatch_toggle_rows(self) -> None:
        app = object.__new__(receiver.PointCloudApp)
        app._visualization_mode = receiver.VISUALIZATION_HEATMAP
        app._detection_enabled = False
        app._object_detection_enabled = False
        app._pose_enabled = False
        app._vlm_enabled = False
        app._scene_description_enabled = False
        app._toggle_scene_description = mock.Mock()

        scene_rect = app._control_panel_row_rects()["scene"]
        scene_click = (
            scene_rect[0] + scene_rect[2] // 2,
            scene_rect[1] + scene_rect[3] // 2,
        )

        self.assertTrue(app._handle_control_panel_click(scene_click))
        app._toggle_scene_description.assert_called_once()

        outside_click = (receiver.CONTROL_PANEL_LEFT_PX - 5, receiver.CONTROL_PANEL_TOP_PX - 5)
        self.assertFalse(app._handle_control_panel_click(outside_click))

    def test_scene_describer_streams_partial_chunks(self) -> None:
        describer = OpenAISceneDescriber(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="test-model",
            timeout_s=1.0,
            max_tokens=32,
            jpeg_quality=80,
            image_max_side_px=128,
        )
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="A "))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="scene."))]),
        ]
        create = mock.Mock(return_value=iter(chunks))
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

        with (
            mock.patch.object(describer, "_encode_rgb_data_url", return_value="data:image/jpeg;base64,x"),
            mock.patch.object(describer, "_ensure_client", return_value=client),
        ):
            parts = list(
                describer.stream_scene(
                    np.zeros((3, 4, 3), dtype=np.uint8),
                    camera_id=2,
                    frame_token=("serial", 1, 2),
                )
            )

        self.assertEqual([part.description for part in parts], ["A", "A scene.", "A scene."])
        self.assertFalse(parts[0].is_final)
        self.assertTrue(parts[-1].is_final)
        self.assertTrue(create.call_args.kwargs["stream"])
        self.assertEqual(create.call_args.kwargs["temperature"], 0.2)

    def test_scene_describer_streams_until_model_finishes(self) -> None:
        describer = OpenAISceneDescriber(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="test-model",
            timeout_s=1.0,
            max_tokens=256,
            jpeg_quality=82,
            image_max_side_px=128,
        )

        class ClosableStream:
            def __init__(self) -> None:
                self.closed = False
                self._chunks = iter(
                    [
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(
                                        content="A person stands beside stacked boxes in a warehouse aisle."
                                    )
                                )
                            ]
                        ),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(
                                        content=" A pallet is nearby with a clear path through the center."
                                    )
                                )
                            ]
                        ),
                        SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(content=" Extra detail should still stream.")
                                )
                            ]
                        ),
                    ]
                )

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._chunks)

            def close(self) -> None:
                self.closed = True

        stream = ClosableStream()
        create = mock.Mock(return_value=stream)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

        with (
            mock.patch.object(describer, "_encode_rgb_data_url", return_value="data:image/jpeg;base64,x"),
            mock.patch.object(describer, "_ensure_client", return_value=client),
        ):
            parts = list(
                describer.stream_scene(
                    np.zeros((3, 4, 3), dtype=np.uint8),
                    camera_id=1,
                    frame_token=("serial", 3, 4),
                )
            )

        self.assertEqual(len(parts), 4)
        self.assertFalse(parts[0].is_final)
        self.assertFalse(parts[1].is_final)
        self.assertFalse(parts[2].is_final)
        self.assertTrue(parts[-1].is_final)
        self.assertEqual(
            parts[-1].description,
            "A person stands beside stacked boxes in a warehouse aisle. A pallet is nearby with a clear path through the center. Extra detail should still stream.",
        )
        self.assertFalse(stream.closed)

    def test_scene_description_worker_streams_all_cameras_in_parallel(self) -> None:
        barrier = threading.Barrier(4)
        started_lock = threading.Lock()
        started = []

        class FakeDescriber:
            def stream_scene(self, color_image, *, camera_id, frame_token):
                with started_lock:
                    started.append(camera_id)
                barrier.wait(timeout=1.0)
                yield SceneDescription(
                    camera_id=camera_id,
                    description=f"camera {camera_id} partial",
                    confidence=1.0,
                    timestamp=time.time(),
                    frame_token=frame_token,
                    latency_ms=1.0,
                    is_final=False,
                )
                yield SceneDescription(
                    camera_id=camera_id,
                    description=f"camera {camera_id} final",
                    confidence=1.0,
                    timestamp=time.time(),
                    frame_token=frame_token,
                    latency_ms=2.0,
                    is_final=True,
                )

        worker = SceneDescriptionWorker(
            receiver=mock.Mock(),
            stop_event=threading.Event(),
            description_cameras=(0, 1, 2, 3),
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="test-model",
            refresh_interval_s=10.0,
            timeout_s=1.0,
            max_tokens=32,
            jpeg_quality=80,
            image_max_side_px=128,
            max_workers=4,
            initially_enabled=True,
            describer_factory=FakeDescriber,
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            worker._run_description_batch(
                executor,
                [self.frame(camera_id) for camera_id in range(4)],
            )

        descriptions = worker.get_descriptions()
        self.assertEqual(set(started), {0, 1, 2, 3})
        self.assertEqual(set(descriptions), {0, 1, 2, 3})
        self.assertTrue(all(description.is_final for description in descriptions.values()))
        self.assertEqual(descriptions[3].description, "camera 3 final")

    def test_scene_description_worker_reschedules_fast_camera_independently(self) -> None:
        frames = {camera_id: self.frame(camera_id) for camera_id in (0, 1)}
        release_slow = threading.Event()
        fast_second_request = threading.Event()
        starts_lock = threading.Lock()
        starts = []

        class FakeReceiver:
            def snapshot_latest_frames(self, camera_ids=None):
                selected_ids = camera_ids if camera_ids is not None else frames.keys()
                return {
                    camera_id: frames[camera_id]
                    for camera_id in selected_ids
                    if camera_id in frames
                }

        class FakeDescriber:
            def stream_scene(self, color_image, *, camera_id, frame_token):
                with starts_lock:
                    starts.append(camera_id)
                    if camera_id == 1 and starts.count(1) >= 2:
                        fast_second_request.set()
                if camera_id == 0:
                    release_slow.wait(timeout=2.0)
                yield SceneDescription(
                    camera_id=camera_id,
                    description=f"camera {camera_id} final",
                    confidence=1.0,
                    timestamp=time.time(),
                    frame_token=frame_token,
                    latency_ms=1.0,
                    is_final=True,
                )

        worker = SceneDescriptionWorker(
            receiver=FakeReceiver(),
            stop_event=threading.Event(),
            description_cameras=(0, 1),
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="test-model",
            refresh_interval_s=0.1,
            timeout_s=1.0,
            max_tokens=32,
            jpeg_quality=80,
            image_max_side_px=128,
            max_workers=2,
            initially_enabled=True,
            describer_factory=FakeDescriber,
        )

        worker.start()
        try:
            self.assertTrue(fast_second_request.wait(timeout=2.0))
            with starts_lock:
                self.assertEqual(starts.count(0), 1)
                self.assertGreaterEqual(starts.count(1), 2)
        finally:
            worker.request_stop()
            release_slow.set()
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())

    def test_latency_probe_echo_records_reported_round_trip_time(self) -> None:
        stop_event = threading.Event()
        frame_receiver = receiver.FrameReceiver("", 0, receiver.Metrics(), stop_event)
        probe_receiver = receiver.LatencyProbeReceiver(
            "127.0.0.1",
            0,
            frame_receiver,
            stop_event,
        )
        probe_receiver.start()
        try:
            port = self.wait_for_udp_port(probe_receiver)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
                client.settimeout(1.0)
                client.sendto(
                    struct.pack(receiver.LATENCY_PROBE_REQUEST_STRUCT, 123, 0),
                    ("127.0.0.1", port),
                )
                echo, _peer = client.recvfrom(receiver.LATENCY_PROBE_RECV_SIZE)
                self.assertEqual(
                    struct.unpack(receiver.LATENCY_PROBE_ECHO_STRUCT, echo),
                    (123,),
                )

                client.sendto(
                    struct.pack(
                        receiver.LATENCY_PROBE_REQUEST_STRUCT,
                        456,
                        2_000_000,
                    ),
                    ("127.0.0.1", port),
                )
                _echo, _peer = client.recvfrom(receiver.LATENCY_PROBE_RECV_SIZE)
                self.wait_for_probe_sample(frame_receiver)

                client.sendto(
                    struct.pack(
                        receiver.LATENCY_PROBE_REQUEST_V2_STRUCT,
                        receiver.LATENCY_PROBE_MAGIC,
                        receiver.LATENCY_PROBE_VERSION,
                        0,
                        7,
                        789,
                        receiver.clock_tai_ns(),
                        3_000_000,
                    ),
                    ("127.0.0.1", port),
                )
                echo, _peer = client.recvfrom(receiver.LATENCY_PROBE_RECV_SIZE)
                self.assertEqual(len(echo), struct.calcsize(receiver.LATENCY_PROBE_ECHO_V2_STRUCT))
                magic, version, _flags, sequence, sender_time_ns, _sender_tai_ns, _receiver_tai_ns = (
                    struct.unpack(receiver.LATENCY_PROBE_ECHO_V2_STRUCT, echo)
                )
                self.assertEqual(magic, receiver.LATENCY_PROBE_MAGIC)
                self.assertEqual(version, receiver.LATENCY_PROBE_VERSION)
                self.assertEqual(sequence, 7)
                self.assertEqual(sender_time_ns, 789)
        finally:
            stop_event.set()
            probe_receiver.stop()
            probe_receiver.join(timeout=2.0)
            self.assertFalse(probe_receiver.is_alive())

    def test_socket_streams_update_and_reconnect_independently(self) -> None:
        stop_event = threading.Event()
        metrics = receiver.Metrics()
        frame_receiver = receiver.FrameReceiver("127.0.0.1", 0, metrics, stop_event)
        clients = []
        frame_receiver.start()
        try:
            port = self.wait_for_port(frame_receiver)
            for camera_id in range(4):
                client = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                clients.append(client)
                self.send_packet(client, self.packet(camera_id, 1))

            self.wait_for_sequences(frame_receiver, {0: 1, 1: 1, 2: 1, 3: 1})
            self.send_packet(clients[0], self.packet(0, 2))
            self.send_packet(clients[2], self.packet(2, 2))
            self.wait_for_sequences(frame_receiver, {0: 2, 1: 1, 2: 2, 3: 1})

            clients[1].close()
            replacement = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            clients.append(replacement)
            self.send_packet(replacement, self.packet(1, 2))
            latest = self.wait_for_sequences(
                frame_receiver,
                {0: 2, 1: 2, 2: 2, 3: 1},
            )
            self.assertEqual(set(latest), {0, 1, 2, 3})
            self.assertGreater(metrics.bitrate_mbps(), 0.0)
        finally:
            for client in clients:
                try:
                    client.close()
                except OSError:
                    pass
            frame_receiver.stop()
            frame_receiver.join(timeout=2.0)
            self.assertFalse(frame_receiver.is_alive())

    def test_detection_cache_clears_only_processed_camera(self) -> None:
        stop_event = threading.Event()
        frame_receiver = receiver.FrameReceiver("", 0, receiver.Metrics(), stop_event)
        worker = receiver.DetectionWorker(
            receiver=frame_receiver,
            stop_event=stop_event,
            detection_cameras=(0, 1),
            model_path="unused",
            conf_threshold=0.5,
            iou_threshold=0.45,
            imgsz=640,
            device="cpu",
            half_precision=False,
            batch_size=1,
            max_det=20,
            retina_masks=False,
            erosion_kernel=0,
            subsample_step=1,
            outlier_threshold_m=0.6,
            min_points=1,
            camera_rotations=[np.eye(3, dtype=np.float32) for _ in range(4)],
            camera_offsets=tuple(np.zeros(3, dtype=np.float32) for _ in range(4)),
            camera_max_depths=(10.0, 10.0, 10.0, 10.0),
            horizontal_fov_deg=90.0,
            vertical_fov_deg=60.0,
        )

        worker._replace_camera_detections((0, 1), [self.box(0), self.box(1)])
        worker._replace_camera_detections((0,), [])
        self.assertEqual([box.camera_id for box in worker.get_detections()], [1])

        worker.set_enabled(True)
        self.assertTrue(worker.is_enabled())
        worker.set_enabled(False)
        self.assertFalse(worker.is_enabled())
        self.assertEqual(worker.get_detections(), [])

    def test_detection_worker_switches_resident_detector_classes(self) -> None:
        stop_event = threading.Event()
        frame_receiver = receiver.FrameReceiver("", 0, receiver.Metrics(), stop_event)
        worker = receiver.DetectionWorker(
            receiver=frame_receiver,
            stop_event=stop_event,
            detection_cameras=(0,),
            model_path="unused",
            conf_threshold=0.5,
            iou_threshold=0.45,
            imgsz=640,
            device="cpu",
            half_precision=False,
            batch_size=1,
            max_det=20,
            retina_masks=False,
            erosion_kernel=0,
            subsample_step=1,
            outlier_threshold_m=0.6,
            min_points=1,
            camera_rotations=[np.eye(3, dtype=np.float32) for _ in range(4)],
            camera_offsets=tuple(np.zeros(3, dtype=np.float32) for _ in range(4)),
            camera_max_depths=(10.0, 10.0, 10.0, 10.0),
            horizontal_fov_deg=90.0,
            vertical_fov_deg=60.0,
        )
        detector = mock.Mock()
        worker._detector = detector
        worker._replace_camera_detections((0,), [self.box(0)])

        worker.set_class_ids(None)
        detector.set_class_ids.assert_called_once_with(None)
        self.assertEqual(worker.get_detections(), [])

    def test_yolo_seg_detector_class_filter_can_switch_to_all_classes(self) -> None:
        detector = YoloSegDetector(class_ids=(0,))
        self.assertEqual(detector._class_ids, (0,))

        detector.set_class_ids(None)
        self.assertIsNone(detector._class_ids)
        self.assertEqual(detector._class_label(mock.Mock(names={56: "chair"}), 56), "chair")

    def test_bbox_label_includes_detection_confidence_and_dimensions(self) -> None:
        box = self.box(0)
        box.confidence = 0.876
        box.size_xyz = np.array([0.45, 1.82, 0.63], dtype=np.float32)

        label = receiver.BBoxRenderer._format_box_label(box)
        self.assertIn("Detected person 88%", label)
        self.assertIn("X 0.45m", label)
        self.assertIn("Y 1.82m", label)
        self.assertIn("Z 0.63m", label)

    def test_vlm_json_validation_clips_boxes_and_caps_objects(self) -> None:
        detections = parse_vlm_detections(
            """
            {
              "objects": [
                {"label": "tool cart", "confidence": 0.85, "bbox_2d": [-5, 100, 800, 900]},
                {"label": "bad reversed", "confidence": 0.90, "bbox_2d": [9, 9, 1, 1]},
                {"label": "too many", "confidence": 0.80, "bbox_2d": [10, 10, 40, 40]}
              ]
            }
            """,
            camera_id=2,
            image_shape=(100, 120, 3),
            frame_timestamp=12.5,
            frame_token=("serial", 7, 8),
            max_objects=1,
            conf_threshold=0.5,
            min_box_side_px=8.0,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "tool cart")
        self.assertEqual(detections[0].camera_id, 2)
        np.testing.assert_allclose(detections[0].bbox_xyxy, [0.0, 10.0, 96.0, 90.0])

        with self.assertRaisesRegex(VLMDetectorError, "invalid VLM JSON"):
            parse_vlm_detections(
                "{",
                camera_id=0,
                image_shape=(10, 10, 3),
                frame_timestamp=0.0,
                frame_token=("", 0, 0),
                max_objects=1,
                conf_threshold=0.0,
                min_box_side_px=1.0,
            )

        self.assertEqual(
            parse_vlm_detections(
                '{"objects":[{"label":"crate","confidence":1.5,"bbox_2d":[0,0,9,9]}]}',
                camera_id=0,
                image_shape=(10, 10, 3),
                frame_timestamp=0.0,
                frame_token=("", 0, 0),
                max_objects=1,
                conf_threshold=0.0,
                min_box_side_px=1.0,
            ),
            [],
        )

    def test_vlm_bbox_only_depth_fit_uses_metric_depth(self) -> None:
        box = fit_bbox_only(
            depth_image=np.full((3, 4), 1000, dtype=np.uint16),
            bbox_xyxy=np.array([0.0, 0.0, 4.0, 3.0], dtype=np.float32),
            camera_id=0,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            camera_rotation_matrix=np.eye(3, dtype=np.float32),
            camera_offset=np.zeros(3, dtype=np.float32),
            erosion_kernel=0,
            subsample_step=1,
            min_points=1,
            label="crate",
            confidence=0.7,
        )

        self.assertIsNotNone(box)
        self.assertEqual(box.label, "crate")
        self.assertEqual(box.source, "vlm")
        self.assertGreaterEqual(box.size_xyz[0], 0.3)

        self.assertIsNone(
            fit_bbox_only(
                depth_image=np.zeros((3, 4), dtype=np.uint16),
                bbox_xyxy=np.array([0.0, 0.0, 4.0, 3.0], dtype=np.float32),
                camera_id=0,
                fx=1.0,
                fy=1.0,
                cx=0.0,
                cy=0.0,
                camera_rotation_matrix=np.eye(3, dtype=np.float32),
                camera_offset=np.zeros(3, dtype=np.float32),
                erosion_kernel=0,
                subsample_step=1,
                min_points=1,
            )
        )

    def test_vlm_fast_line_parser_scales_boxes_and_rejects_none(self) -> None:
        detections = parse_vlm_line_detections(
            "crate|0.875|100|200|900|800\nnone|0|0|0|0|0",
            camera_id=1,
            image_shape=(100, 200, 3),
            frame_timestamp=5.0,
            frame_token=("serial", 3, 4),
            max_objects=2,
            conf_threshold=0.5,
            min_box_side_px=1.0,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "crate")
        np.testing.assert_allclose(detections[0].bbox_xyxy, [20.0, 20.0, 180.0, 80.0])

    def test_vlm_worker_round_robin_and_stale_cache(self) -> None:
        worker = self.vlm_worker((0, 1))
        latest = {0: self.frame(0, 1), 1: self.frame(1, 1)}
        last_tokens = {}

        frame, cursor = worker._select_next_changed_frame(latest, last_tokens, 0)
        self.assertEqual(frame.camera_id, 0)
        last_tokens[0] = worker._frame_token(frame)
        frame, cursor = worker._select_next_changed_frame(latest, last_tokens, cursor)
        self.assertEqual(frame.camera_id, 1)

        worker._replace_camera_detections(0, [self.box(0)], success_monotonic_s=0.0)
        worker._replace_camera_detections(1, [self.box(1)])
        self.assertEqual([box.camera_id for box in worker.get_detections()], [1])

        worker._replace_camera_detections(0, [self.box(0)])
        worker._replace_camera_detections(0, [])
        self.assertEqual([box.camera_id for box in worker.get_detections()], [1])

    def test_vlm_camera_shards_keep_request_workers_disjoint(self) -> None:
        self.assertEqual(
            receiver._split_vlm_camera_groups((0, 1, 2, 3), 2),
            [(0, 2), (1, 3)],
        )
        self.assertEqual(receiver._split_vlm_camera_groups((0,), 2), [(0,)])

    def test_vlm_bbox_label_is_source_distinct(self) -> None:
        box = self.box(0)
        box.source = "vlm"
        box.label = "crate"

        label = receiver.BBoxRenderer._format_box_label(box)
        self.assertIn("VLM crate", label)
        self.assertNotEqual(
            receiver.BBoxRenderer._make_box_signature([box]),
            receiver.BBoxRenderer._make_box_signature([self.box(0)]),
        )

    def test_pose_backprojection_rejects_out_of_frame_keypoints(self) -> None:
        depth = np.full((3, 4), 1000, dtype=np.uint16)
        keypoints_3d, valid = backproject_keypoints(
            keypoints_xy=np.array([[-1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            keypoints_confidence=np.array([1.0, 1.0], dtype=np.float32),
            depth_image=depth,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            camera_rotation_matrix=np.eye(3, dtype=np.float32),
            camera_offset=np.zeros(3, dtype=np.float32),
            depth_sample_radius=0,
        )

        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(np.isnan(keypoints_3d[0]).all())
        np.testing.assert_allclose(keypoints_3d[1], [1.0, -1.0, 1.0])

    def test_skeleton_signature_caches_invalid_nan_keypoints(self) -> None:
        pose = mock.Mock()
        pose.camera_id = 0
        pose.label = "person"
        pose.confidence = 0.75
        pose.timestamp = 1.0
        pose.center_xyz = np.zeros(3, dtype=np.float32)
        pose.size_xyz = np.ones(3, dtype=np.float32)
        pose.keypoints_3d = np.full((17, 3), np.nan, dtype=np.float32)
        pose.keypoints_3d[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        pose.keypoints_confidence = np.ones(17, dtype=np.float32)

        signature = receiver.BBoxRenderer._make_skeleton_signature([pose])
        second_signature = receiver.BBoxRenderer._make_skeleton_signature([pose])

        self.assertEqual(signature, second_signature)

    def test_pose_fit_rejects_depth_points_without_skeleton_connections(self) -> None:
        keypoints_xy = np.zeros((17, 2), dtype=np.float32)
        keypoints_confidence = np.zeros(17, dtype=np.float32)
        for keypoint_idx, pixel in zip((3, 4, 9, 10), ((1, 1), (2, 1), (1, 2), (2, 2))):
            keypoints_xy[keypoint_idx] = pixel
            keypoints_confidence[keypoint_idx] = 1.0

        pose = fit_pose_bbox(
            depth_image=np.full((4, 4), 1000, dtype=np.uint16),
            keypoints_xy=keypoints_xy,
            keypoints_confidence=keypoints_confidence,
            camera_id=0,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            camera_rotation_matrix=np.eye(3, dtype=np.float32),
            camera_offset=np.zeros(3, dtype=np.float32),
            confidence_threshold=0.5,
            min_valid_keypoints=4,
            min_valid_connections=1,
        )

        self.assertIsNone(pose)

    def test_bbox_label_anchor_holds_through_box_jitter(self) -> None:
        anchor = receiver.BBoxRenderer._stabilize_label_anchor(
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([0.05, 1.02, 1.96], dtype=np.float32),
        )

        np.testing.assert_allclose(anchor, [0.0, 1.0, 2.0], atol=1e-6)

    def test_bbox_label_anchor_repositions_after_real_motion(self) -> None:
        anchor = receiver.BBoxRenderer._stabilize_label_anchor(
            np.array([0.0, 1.0, 2.0], dtype=np.float32),
            np.array([0.0, 1.0, 2.5], dtype=np.float32),
        )

        np.testing.assert_allclose(anchor, [0.0, 1.0, 2.5], atol=1e-6)

    def test_bbox_extent_trim_ignores_sparse_edge_outlier(self) -> None:
        cluster = np.column_stack(
            (
                np.linspace(0.0, 1.0, 100, dtype=np.float32),
                np.zeros(100, dtype=np.float32),
                np.ones(100, dtype=np.float32),
            )
        )
        points = np.vstack((cluster, np.array([[100.0, 0.0, 1.0]], dtype=np.float32)))

        _raw_mins, raw_maxs = _fit_axis_aligned_extents(points, trim_percentile=0.0)
        _trimmed_mins, trimmed_maxs = _fit_axis_aligned_extents(
            points,
            trim_percentile=1.0,
        )

        self.assertEqual(float(raw_maxs[0]), 100.0)
        self.assertLessEqual(float(trimmed_maxs[0]), 1.0)

    @staticmethod
    def box(camera_id: int) -> receiver.PersonBBox3D:
        return receiver.PersonBBox3D(
            camera_id=camera_id,
            center_xyz=np.zeros(3, dtype=np.float32),
            size_xyz=np.ones(3, dtype=np.float32),
            confidence=1.0,
            label="person",
            num_points=1,
            timestamp=0.0,
        )

    @staticmethod
    def vlm_worker(camera_ids: tuple[int, ...]) -> receiver.VLMDetectionWorker:
        return receiver.VLMDetectionWorker(
            receiver=mock.Mock(),
            stop_event=threading.Event(),
            detection_cameras=camera_ids,
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model="test-model",
            request_rate_hz=2.0,
            timeout_s=1.0,
            max_tokens=32,
            jpeg_quality=80,
            image_max_side_px=640,
            max_objects=2,
            conf_threshold=0.5,
            min_box_side_px=1.0,
            stale_ttl_s=1.0,
            erosion_kernel=0,
            subsample_step=1,
            outlier_threshold_m=0.6,
            min_points=1,
            camera_rotations=[np.eye(3, dtype=np.float32) for _ in range(4)],
            camera_offsets=tuple(np.zeros(3, dtype=np.float32) for _ in range(4)),
            camera_max_depths=(10.0, 10.0, 10.0, 10.0),
            horizontal_fov_deg=90.0,
            vertical_fov_deg=60.0,
        )

    @staticmethod
    def send_packet(client: socket.socket, packet: dict) -> None:
        payload = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        client.sendall(struct.pack(receiver.MESSAGE_HEADER_STRUCT, len(payload)))
        client.sendall(payload)

    @staticmethod
    def wait_for_port(frame_receiver: receiver.FrameReceiver) -> int:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            server = frame_receiver._server_socket
            if server is not None:
                return int(server.getsockname()[1])
            time.sleep(0.01)
        raise AssertionError("receiver server did not start")

    @staticmethod
    def wait_for_udp_port(probe_receiver: receiver.LatencyProbeReceiver) -> int:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            sock = probe_receiver._socket
            if sock is not None:
                return int(sock.getsockname()[1])
            time.sleep(0.01)
        raise AssertionError("latency probe receiver did not start")

    @staticmethod
    def wait_for_probe_sample(frame_receiver: receiver.FrameReceiver) -> None:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            summary = frame_receiver.latency_summary()
            if summary["count"]:
                return
            time.sleep(0.01)
        raise AssertionError("latency probe sample was not recorded")

    @staticmethod
    def wait_for_sequences(
        frame_receiver: receiver.FrameReceiver,
        expected: dict[int, int],
    ) -> dict[int, receiver.CameraFrame]:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            latest = frame_receiver.snapshot_latest_frames()
            if all(
                latest.get(camera_id) is not None
                and latest[camera_id].sequence == sequence
                for camera_id, sequence in expected.items()
            ):
                return latest
            time.sleep(0.01)
        raise AssertionError(f"sequences did not reach {expected}")


if __name__ == "__main__":
    unittest.main()
