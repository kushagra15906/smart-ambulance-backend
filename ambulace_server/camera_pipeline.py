"""
camera_pipeline.py
==================
Lives in:  ambulance_server/camera_pipeline.py

Connects ESP32-CAM raw frames to your existing DualYOLODetector.

Interface used from yolo_detector.py:
    from yolo_detector import detector          ← singleton DualYOLODetector
    detector.load_models()                      ← call once at startup
    result = detector.detect_all(rgb_frame)     ← run both models
    result = detector.detect_ambulance(frame)   ← ambulance only
    result = detector.detect_vehicles(frame)    ← vehicle count only

    detect_ambulance() returns:
        {
            "detected"       : bool,
            "confidence"     : float,
            "bbox"           : [x1, y1, x2, y2] or [],
            "annotated_frame": np.ndarray,
        }

    detect_vehicles() returns:
        {
            "total_count"    : int,
            "by_class"       : {"car": 2, "truck": 1, ...},
            "detections"     : [...],
            "annotated_frame": np.ndarray,
        }

    detect_all() returns both merged together.
"""

import logging
import threading
import time
import queue
from typing import Optional, Tuple

import cv2
import numpy as np

# ── Import your existing DualYOLODetector singleton ──────────────────────────
# detector is the singleton defined at the bottom of yolo_detector.py
from yolo_detector import detector as _yolo_detector

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
YOLO_DETECT_EVERY  = 5     # run detection every Nth frame
                            # at ~7 fps → ~1.4 detections per second
DETECTION_HOLD_SEC = 8     # keep detected=True for 8s after last positive
                            # prevents flickering between GREEN/RED
AMBULANCE_CONF_MIN = 0.45  # match your CONF_THRESH in yolo_detector.py


# ═══════════════════════════════════════════════════════════════════════════════
#  CameraChannel — one ESP32-CAM per intersection
# ═══════════════════════════════════════════════════════════════════════════════

class CameraChannel:
    """
    One instance per signal intersection.
    Accepts raw grayscale frames from ESP32-CAM.
    Runs DualYOLODetector in background thread.
    Builds JPEG for MJPEG /stream endpoint.
    """

    def __init__(self, signal_id: str):
        self.signal_id        = signal_id
        self._frame_count     = 0
        self.total_frames     = 0
        self.total_detections = 0
        self.last_vehicle_count = 0

        # Latest JPEG bytes for /stream
        self._latest_jpeg = None
        self._jpeg_lock   = threading.Lock()

        # Detection state
        self._detected      = False
        self._confidence    = 0.0
        self._last_seen_ts  = 0.0
        self._det_lock      = threading.Lock()

        # Background YOLO work queue — maxsize=1 so we always process latest frame
        self._detect_queue = queue.Queue(maxsize=1)

        # Start background worker thread
        threading.Thread(
            target=self._yolo_worker,
            daemon=True,
            name=f"yolo-{signal_id}"
        ).start()

        log.info("[CAM] Channel ready: %s", signal_id)

    # ── Called by POST /stream-frame in app.py ────────────────────────────────

    def ingest_frame(self, raw_bytes: bytes, width: int, height: int) -> dict:
        """
        Process one raw grayscale frame from ESP32-CAM.
        - Decodes bytes → numpy array
        - Queues for YOLO every Nth frame
        - Builds annotated JPEG for /stream
        Returns dict for HTTP response back to ESP32-CAM.
        """
        self._frame_count += 1
        self.total_frames  += 1

        # Validate
        expected = width * height
        if len(raw_bytes) != expected:
            log.warning("[CAM] %s size mismatch: got %d expected %d",
                        self.signal_id, len(raw_bytes), expected)
            return {
                "error"     : "size_mismatch",
                "got"       : len(raw_bytes),
                "expected"  : expected,
                "detected"  : False,
                "confidence": 0.0,
            }

        # Decode grayscale bytes → 2D numpy array
        gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width))

        # Queue for YOLO every Nth frame (non-blocking — drop if worker busy)
        if self._frame_count % YOLO_DETECT_EVERY == 0:
            # Convert gray → BGR here (DualYOLODetector expects BGR/RGB)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            try:
                self._detect_queue.put_nowait(bgr)
            except queue.Full:
                pass  # worker still busy — skip, get next frame

        # Build JPEG for /stream (every frame, lightweight)
        self._build_stream_jpeg(gray, width, height)

        detected, conf = self.is_detected()
        return {
            "frame_no"    : self._frame_count,
            "detected"    : detected,
            "confidence"  : round(conf, 3),
            "vehicle_count": self.last_vehicle_count,
        }

    # ── Called by GET /signal in app.py ──────────────────────────────────────

    def is_detected(self) -> Tuple[bool, float]:
        """
        Returns (detected: bool, confidence: float).
        Holds detected=True for DETECTION_HOLD_SEC after last sighting.
        This prevents the traffic signal from flickering every frame.
        """
        with self._det_lock:
            if self._detected:
                return True, self._confidence
            # Hold period — gradually fade confidence during hold
            if self._last_seen_ts > 0:
                elapsed = time.time() - self._last_seen_ts
                if elapsed < DETECTION_HOLD_SEC:
                    held = max(0.0, self._confidence * (1.0 - elapsed / DETECTION_HOLD_SEC))
                    return True, round(held, 3)
            return False, 0.0

    # ── Called by GET /stream in app.py ──────────────────────────────────────

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns latest annotated JPEG for MJPEG streaming."""
        with self._jpeg_lock:
            return self._latest_jpeg

    # ── Background YOLO worker thread ─────────────────────────────────────────

    def _yolo_worker(self):
        """
        Runs continuously in background.
        Pulls BGR frames from queue → runs DualYOLODetector.detect_all()
        → updates detection state and vehicle count.
        Never blocks the HTTP request threads.
        """
        while True:
            try:
                bgr = self._detect_queue.get(timeout=1.0)

                # ── Run detect_all() — uses both your YOLO models ─────────────
                # detect_all() returns:
                #   vehicle_count, vehicles_by_class, vehicle_detections,
                #   ambulance_detected, ambulance_conf, ambulance_bbox,
                #   annotated_frame, timestamp
                result = _yolo_detector.detect_all(bgr)

                ambulance_detected = result.get("ambulance_detected", False)
                ambulance_conf     = result.get("ambulance_conf",     0.0)
                vehicle_count      = result.get("vehicle_count",      0)
                annotated_frame    = result.get("annotated_frame")

                # Update detection state
                with self._det_lock:
                    self._detected   = ambulance_detected
                    self._confidence = ambulance_conf
                    if ambulance_detected:
                        self._last_seen_ts = time.time()
                        self.total_detections += 1
                        log.info("[YOLO] 🚨 AMBULANCE at %s  conf=%.2f",
                                 self.signal_id, ambulance_conf)

                # Update vehicle count (used by /traffic endpoint in app.py)
                self.last_vehicle_count = vehicle_count

                # Update JPEG stream with YOLO-annotated frame
                # annotated_frame already has bounding boxes drawn by yolo_detector.py
                if annotated_frame is not None:
                    self._update_jpeg_from_annotated(annotated_frame, ambulance_detected, ambulance_conf)

            except queue.Empty:
                # No frame queued — just loop
                # Also auto-clear detection state if hold expired
                with self._det_lock:
                    if (self._detected and self._last_seen_ts > 0 and
                            time.time() - self._last_seen_ts > DETECTION_HOLD_SEC):
                        self._detected = False
                        log.info("[CAM] %s detection hold expired → cleared", self.signal_id)

            except Exception as e:
                log.error("[YOLO] Worker error at %s: %s", self.signal_id, e)

    # ── JPEG builders ─────────────────────────────────────────────────────────

    def _build_stream_jpeg(self, gray: np.ndarray, w: int, h: int):
        """
        Fast JPEG build for every non-YOLO frame.
        Simple grayscale → BGR conversion + basic overlay.
        Called every frame to keep stream smooth.
        """
        display = cv2.resize(gray, (480, 360), interpolation=cv2.INTER_LINEAR)
        bgr     = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._add_overlay(bgr)
        self._encode_and_store(bgr)

    def _update_jpeg_from_annotated(self, annotated: np.ndarray,
                                     detected: bool, conf: float):
        """
        Use the annotated frame from DualYOLODetector (already has bboxes).
        Scales to stream size and adds our overlay.
        """
        try:
            display = cv2.resize(annotated, (480, 360),
                                 interpolation=cv2.INTER_LINEAR)
            self._add_overlay(display, detected, conf)
            self._encode_and_store(display)
        except Exception as e:
            log.error("[CAM] JPEG update error: %s", e)

    def _add_overlay(self, bgr: np.ndarray,
                     detected: bool = None, conf: float = 0.0):
        """Add signal ID, status text, frame counter overlay."""
        if detected is None:
            detected, conf = self.is_detected()

        # Top-left: camera ID
        cv2.putText(bgr, f"Cam: {self.signal_id}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1)

        # Top-right: vehicle count
        cv2.putText(bgr, f"Vehicles: {self.last_vehicle_count}",
                    (300, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (180, 255, 180), 1)

        # Bottom: ambulance status
        status_text  = f"AMBULANCE {conf:.0%}" if detected else "Monitoring..."
        status_color = (0, 60, 255) if detected else (0, 220, 0)
        cv2.putText(bgr, status_text,
                    (8, 348), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, status_color, 2)

        # Frame counter
        cv2.putText(bgr, f"#{self._frame_count}",
                    (420, 348), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (100, 100, 100), 1)

    def _encode_and_store(self, bgr: np.ndarray):
        """JPEG encode and store as latest frame."""
        ret, buf = cv2.imencode(".jpg", bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ret:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()


# ═══════════════════════════════════════════════════════════════════════════════
#  MultiCameraManager — manages all channels, imported by app.py
# ═══════════════════════════════════════════════════════════════════════════════

class MultiCameraManager:
    """
    Central manager for all ESP32-CAM channels.
    Imported and used by app.py exactly as before.

    On init:
      1. Calls detector.load_models() once — loads both YOLO models into RAM
      2. Creates one CameraChannel per signal config

    All channels share the SAME DualYOLODetector singleton from yolo_detector.py.
    One model in memory — not one copy per camera.
    """

    def __init__(self, signal_configs: list, history_manager=None):
        self._channels : dict[str, CameraChannel] = {}
        self._history  = history_manager

        # ── Load YOLO models once at startup ──────────────────────────────────
        log.info("[CAM] Loading YOLO models...")
        loaded = _yolo_detector.load_models()
        if loaded:
            log.info("[CAM] YOLO models ready ✓")
        else:
            log.warning("[CAM] YOLO models failed to load — "
                        "detection will return empty results")

        # ── Create channel per signal ─────────────────────────────────────────
        for cfg in signal_configs:
            sig_id = cfg.get("signal_id")
            if sig_id:
                self._channels[sig_id] = CameraChannel(signal_id=sig_id)

        log.info("[CAM] MultiCameraManager ready — channels: %s",
                 list(self._channels.keys()))

    def start_all(self):
        """No-op — threads already started in CameraChannel.__init__.
        Kept so app.py doesn't need to change."""
        log.info("[CAM] All camera channels active")

    def ingest_frame(self, signal_id: str,
                     raw_bytes: bytes, width: int, height: int) -> dict:
        """
        Called by POST /stream-frame in app.py.
        Auto-creates channel if new signal_id seen.
        """
        if signal_id not in self._channels:
            log.info("[CAM] Auto-creating channel: %s", signal_id)
            self._channels[signal_id] = CameraChannel(signal_id=signal_id)
        return self._channels[signal_id].ingest_frame(raw_bytes, width, height)

    def is_ambulance_detected(self) -> Tuple[bool, Optional[str]]:
        """
        Called by GET /signal in app.py.
        Returns (detected: bool, signal_id: str or None).
        Checks all channels — returns first positive hit.
        """
        for sig_id, ch in self._channels.items():
            detected, conf = ch.is_detected()
            if detected:
                return True, sig_id
        return False, None

    def get_latest_jpeg(self, signal_id: Optional[str] = None) -> Optional[bytes]:
        """
        Called by GET /stream in app.py.
        Returns JPEG bytes for MJPEG streaming.
        """
        if signal_id and signal_id in self._channels:
            return self._channels[signal_id].get_latest_jpeg()
        # No specific ID → first available
        for ch in self._channels.values():
            frame = ch.get_latest_jpeg()
            if frame:
                return frame
        return None

    def get_vehicle_counts(self) -> dict:
        """
        Returns latest vehicle count per channel.
        Used by /traffic endpoint in app.py to update vehicle counts.
        """
        return {
            sig_id: ch.last_vehicle_count
            for sig_id, ch in self._channels.items()
        }

    def get_stats(self) -> dict:
        """Called by /status endpoint in app.py."""
        return {
            sig_id: {
                "total_frames"      : ch.total_frames,
                "total_detections"  : ch.total_detections,
                "currently_detected": ch.is_detected()[0],
                "confidence"        : ch.is_detected()[1],
                "vehicle_count"     : ch.last_vehicle_count,
            }
            for sig_id, ch in self._channels.items()
        }
