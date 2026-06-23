#!/usr/bin/env python3
"""
traffic_controller.py
=====================
Integrates three components into one runnable script:

  1.  Arducam multi-camera cycling  (hardware logic from stream_camera_arducam.py)
  2.  YOLOv8 ONNX vehicle detection  (OpenCV DNN backend — no ultralytics needed)
  3.  Monte Carlo green-time optimiser  (CRN-enhanced, ported from stats_comparison.py)

Data flow per full camera round
────────────────────────────────
  Camera B active → grab frames every DETECTION_INTERVAL s → run YOLO
  Camera C active → grab frames …
  Camera D active → grab frames …
         ↓
  queue_counts = [median_B, median_C, median_D]
  best_split, score = mc_best_split_crn(queue_counts)
         ↓
  Print optimal green-time split
  → (next step) send_to_arduino(best_split)

The MJPEG dashboard (port 8000) keeps running throughout for monitoring.

Dependencies (beyond what the Pi already has):
  pip install opencv-python numpy
  # RPi.GPIO and picamera2 are assumed already installed.
"""

import io
import logging
import random as _random
import socketserver
from http import server
from threading import Condition, Thread
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as gp
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

# generate_candidate_splits is pure math (no randomness) — safe to reuse.
# We do NOT import monte_carlo_best_split because we use the CRN-enhanced
# version defined below.
from monte_carlo_core import generate_candidate_splits

# ══════════════════════════════ CONFIG ════════════════════════════════════════

RESOLUTION          = (640, 480)
HTTP_PORT           = 8000
CAMERAS_TO_CYCLE    = ["B", "C", "D"]      # one camera = one traffic lane
                                            # keep exactly 3 — generate_candidate_splits
                                            # is hardcoded for 3 lanes

# Camera timing
CYCLE_TIME          = 5      # seconds each camera stays active per round
DETECTION_INTERVAL  = 1.0    # seconds between YOLO grabs within each window

# YOLO  ← set YOLO_MODEL_PATH to wherever your .onnx lives
YOLO_MODEL_PATH     = "best (1).onnx"
YOLO_INPUT_SIZE     = (320, 320)
CONF_THRESHOLD      = 0.40
# COCO class IDs that count as "vehicles"
VEHICLE_CLASS_IDS   = frozenset({2, 3, 5, 7})   # car, motorcycle, bus, truck

# Monte Carlo
MC_CYCLE_BUDGET     = 60    # total green-time budget per signal cycle (seconds)
MC_MIN_GREEN        = 5     # minimum green time per lane (seconds)
MC_SECONDS_PER_CAR  = 2.0   # seconds of green to clear one queued car
MC_TRIALS           = 300   # pre-generated arrival scenarios (CRN pool size)
MC_STEP             = 5     # search-step size for candidate generation (seconds)

# Arrival rates: avg new cars arriving per second, per lane.
# These are independent of queue depth — cars keep arriving on red too.
# Tune these to match your actual junction; non-uniform is realistic.
# (stats_comparison.py used [0.12, 0.18, 0.08] as an example.)
MC_ARRIVAL_RATE_PER_LANE = [0.12, 0.18, 0.08]   # [lane_B, lane_C, lane_D]

# ══════════════════════════════════════════════════════════════════════════════


# ───────────────────── CRN-ENHANCED MONTE CARLO ───────────────────────────────
#
# Ported from stats_comparison.py's pregenerate_trial_arrivals() pattern.
#
# WHY CRN MATTERS HERE:
#   monte_carlo_core.py draws fresh random arrivals for every single candidate
#   split it evaluates.  That means two splits can score similarly purely
#   because one got lucky with low arrivals — not because they're actually
#   similar in quality.  With only 300 trials this noise is meaningful.
#
#   Common Random Numbers (CRN) fixes this: we generate ALL arrival outcomes
#   ONCE and replay that exact same set for every candidate.  Now the only
#   thing that differs between candidates' scores is how well they clear the
#   existing queue — which is exactly what we want to optimise.

def _poisson_crn(lam: float, rng: _random.Random) -> int:
    """Knuth's Poisson sampler with an explicit RNG for CRN control."""
    L = pow(2.718281828, -lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _pregenerate_arrivals(
    num_trials: int,
    arrival_rate_per_lane: list[float],
    cycle_budget: int,
    rng: _random.Random,
) -> list[list[int]]:
    """
    Pre-generate num_trials sets of random arrival counts (one per lane).
    Called ONCE per decision; the result is replayed across every candidate.
    """
    return [
        [
            _poisson_crn(arrival_rate_per_lane[lane] * cycle_budget, rng)
            for lane in range(len(arrival_rate_per_lane))
        ]
        for _ in range(num_trials)
    ]


def _score_split(queue_lengths, green_split, shared_arrivals, seconds_per_car):
    """Average residual cars across the pre-generated arrival scenarios."""
    total = 0.0
    for arrivals in shared_arrivals:
        for lane in range(len(queue_lengths)):
            cleared = min(queue_lengths[lane], green_split[lane] / seconds_per_car)
            total  += (queue_lengths[lane] - cleared) + arrivals[lane]
    return total / len(shared_arrivals)


def mc_best_split_crn(
    queue_lengths: list[int],
    cycle_budget_seconds: int      = MC_CYCLE_BUDGET,
    min_green_seconds: int         = MC_MIN_GREEN,
    seconds_per_car: float         = MC_SECONDS_PER_CAR,
    arrival_rate_per_lane: list    = None,
    trials_per_candidate: int      = MC_TRIALS,
    search_step_seconds: int       = MC_STEP,
) -> tuple[list[int], float]:
    """
    CRN-enhanced Monte Carlo green-split optimiser.

    Returns (best_split, best_score) where best_score is the average
    residual car count under the optimal split.
    """
    if arrival_rate_per_lane is None:
        arrival_rate_per_lane = MC_ARRIVAL_RATE_PER_LANE

    rng = _random.Random()   # fresh, unseeded — not shared with anything else

    candidates = generate_candidate_splits(
        len(queue_lengths), cycle_budget_seconds,
        min_green_seconds, search_step_seconds,
    )

    # THE KEY STEP: one shared pool of arrivals, replayed for every candidate.
    shared_arrivals = _pregenerate_arrivals(
        trials_per_candidate, arrival_rate_per_lane, cycle_budget_seconds, rng
    )

    best_split, best_score = None, float("inf")
    for split in candidates:
        score = _score_split(queue_lengths, split, shared_arrivals, seconds_per_car)
        if score < best_score:
            best_score, best_split = score, split

    return best_split, best_score

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────── YOLO DETECTOR ───────────────────────────────────

class YOLOVehicleDetector:
    """
    Uses onnxruntime to load a YOLOv8-exported ONNX model and count vehicles
    in a single BGR frame. onnxruntime handles the DFL reshape layer that
    OpenCV DNN cannot.

    YOLOv8 ONNX output layout (post-export):
        shape  → (1, 84, 8400)
        cols 0-3  → bbox (cx, cy, w, h)   — not needed for counting
        cols 4-83 → class scores for 80 COCO classes
    """

    def __init__(
        self,
        model_path: str         = YOLO_MODEL_PATH,
        conf_threshold: float   = CONF_THRESHOLD,
        input_size: tuple       = YOLO_INPUT_SIZE,
        vehicle_class_ids       = VEHICLE_CLASS_IDS,
    ):
        print(f"[YOLO] Loading model: {model_path}")
        self.session         = ort.InferenceSession(model_path)
        self.input_name      = self.session.get_inputs()[0].name
        self.conf_threshold  = conf_threshold
        self.input_size      = input_size
        self._vehicle_ids    = np.array(list(vehicle_class_ids), dtype=np.int32)
        print("[YOLO] Model ready")

    def count_vehicles(self, bgr_frame) -> int: 
        """
        Run inference on one BGR frame.
        Returns the integer number of vehicle detections above the
        confidence threshold.  Returns 0 on bad / empty input.
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return 0

        img = cv2.resize(bgr_frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, 0)          # add batch dim

        raw = self.session.run(None, {self.input_name: img})[0]  # (1, 84, 8400)

        preds      = raw[0].T             # (8400, 84)
        cls_scores = preds[:, 4:]         # (8400, 80)  — skip bbox cols
        max_scores = cls_scores.max(axis=1)
        pred_cls   = cls_scores.argmax(axis=1)

        conf_ok    = max_scores >= self.conf_threshold
        vehicle_ok = np.isin(pred_cls, self._vehicle_ids)
        return int((conf_ok & vehicle_ok).sum())


# ─────────────────────── MJPEG STREAMING LAYER ───────────────────────────────

class StreamingOutput(io.BufferedIOBase):
    """
    Thread-safe ring buffer that always holds the latest JPEG frame.
    Picamera2 writes into it; the HTTP handler and detector both read from it.
    """

    def __init__(self):
        self.frame     = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

    def latest_frame(self):
        """Non-blocking: return the most recent JPEG bytes, or None."""
        with self.condition:
            return self.frame


# One persistent buffer per camera.
outputs: dict[str, StreamingOutput] = {cam: StreamingOutput() for cam in CAMERAS_TO_CYCLE}


def _dashboard_html(cameras):
    rows = "".join(
        f'<div class="box"><h3>LANE {c}</h3>'
        f'<img src="stream_{c}.mjpg" alt="Camera {c}"></div>'
        for c in cameras
    )
    return (
        "<html><head><title>Traffic Dashboard</title>"
        "<style>"
        "body{background:#111;color:#eee;font-family:sans-serif;margin:0}"
        "h2{text-align:center;padding:18px 0;letter-spacing:2px}"
        ".dash{display:flex;flex-wrap:wrap;justify-content:center;gap:24px;padding:16px}"
        ".box{background:#1c1c1c;padding:14px;border-radius:8px;"
        "     border:1px solid #333;text-align:center}"
        "h3{color:#4CAF50;margin:0 0 10px}"
        "img{display:block;width:100%;max-width:560px;border-radius:4px;background:#000}"
        "</style></head><body>"
        f"<h2>TRAFFIC DASHBOARD — AUTO CYCLING</h2>"
        f"<div class='dash'>{rows}</div>"
        "</body></html>"
    )


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = _dashboard_html(CAMERAS_TO_CYCLE).encode()
            self._send(200, "text/html", body)

        elif self.path.startswith("/stream_") and self.path.endswith(".mjpg"):
            cam_id = self.path[8:-5]        # strip "/stream_" and ".mjpg"
            if cam_id in CAMERAS_TO_CYCLE:
                self._serve_mjpeg(cam_id)
            else:
                self.send_error(404); self.end_headers()
        else:
            self.send_error(404); self.end_headers()

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self, cam_id):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        buf = outputs[cam_id]
        try:
            while True:
                with buf.condition:
                    buf.condition.wait()
                    frame = buf.frame
                if frame:
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            logging.debug(f"Client disconnected from {cam_id}")

    def log_message(self, *_):
        pass   # suppress per-request console noise


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads      = True


# ─────────────────────────── HARDWARE HELPERS ────────────────────────────────

_CAM_HW = {
    "A": {"i2c": "0x04", "pins": (False, False, True)},
    "B": {"i2c": "0x05", "pins": (True,  False, True)},
    "C": {"i2c": "0x06", "pins": (False, True,  False)},
    "D": {"i2c": "0x07", "pins": (True,  True,  False)},
}


def setup_gpio():
    gp.setwarnings(False)
    gp.setmode(gp.BOARD)
    for pin in (7, 11, 12):
        gp.setup(pin, gp.OUT)


def activate_camera(cam_id: str):
    """Select the given camera port via I²C + GPIO multiplexer."""
    hw = _CAM_HW[cam_id.upper()]
    os.system(f"sudo i2cset -y 10 0x70 0x00 {hw['i2c']}")
    gp.output(7,  hw["pins"][0])
    gp.output(11, hw["pins"][1])
    gp.output(12, hw["pins"][2])
    time.sleep(1.5)   # electrical + sensor warm-up


def jpeg_to_bgr(jpeg_bytes: bytes):
    """Decode JPEG bytes (from StreamingOutput) into a BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ─────────────────────── MAIN CAMERA + DETECTION LOOP ────────────────────────

def camera_and_detection_loop(detector: YOLOVehicleDetector):
    """
    Runs forever.  One iteration = one full round over all cameras.

    For each camera
    ───────────────
    • activate hardware
    • start MJPEG recording into the shared StreamingOutput
    • every DETECTION_INTERVAL seconds: grab the latest frame, run YOLO,
      store the count
    • after CYCLE_TIME seconds: stop recording, apply cooldown

    After all cameras complete
    ─────────────────────────
    • queue_counts = [median count per lane]
    • run mc_best_split_crn(queue_counts)
    • print optimal green-time split
    • call send_to_arduino(best_split)  ← stubbed, wired in next step
    """
    while True:
        queue_counts: list[int] = []

        for cam_id in CAMERAS_TO_CYCLE:
            print(f"\n[CAM] ── Activating camera {cam_id} ──")
            activate_camera(cam_id)

            buf            = outputs[cam_id]
            sample_counts: list[int] = []
            picam2         = Picamera2()

            try:
                picam2.configure(
                    picam2.create_video_configuration(main={"size": RESOLUTION})
                )
                picam2.start_recording(MJPEGEncoder(), FileOutput(buf))

                deadline = time.time() + CYCLE_TIME
                while time.time() < deadline:
                    jpeg = buf.latest_frame()
                    if jpeg is not None:
                        bgr   = jpeg_to_bgr(jpeg)
                        count = detector.count_vehicles(bgr)
                        sample_counts.append(count)
                        print(f"  [YOLO] cam {cam_id}: {count} vehicle(s) detected")
                    time.sleep(DETECTION_INTERVAL)

            except Exception as exc:
                print(f"[ERROR] Camera {cam_id}: {exc}")
                time.sleep(2)

            finally:
                try:
                    picam2.stop_recording()
                    picam2.close()
                except Exception:
                    pass
                time.sleep(1.0)   # wait for kernel to release /dev/video0

            # Use median to filter out single-frame outliers
            lane_queue = int(np.median(sample_counts)) if sample_counts else 0
            print(f"[LANE] {cam_id} → queue estimate: {lane_queue} vehicle(s)")
            queue_counts.append(lane_queue)

        # ── Monte Carlo optimisation (CRN-enhanced) ───────────────────────
        print(f"\n[MC] Running CRN optimiser — lane queues: {queue_counts}")
        best_split, best_score = mc_best_split_crn(queue_counts)

        if best_split is None:
            print("[MC] No valid split found — check MC_CYCLE_BUDGET / MC_MIN_GREEN.")
            continue

        print(f"[MC] Optimal split: {best_split}  "
              f"(avg residual cars: {best_score:.2f})")
        for cam_id, green_s in zip(CAMERAS_TO_CYCLE, best_split):
            print(f"     Lane {cam_id} → {green_s}s green")

        # ── TODO (next step): send timing to Arduino ──────────────────────
        # send_to_arduino(best_split)
        # ─────────────────────────────────────────────────────────────────


# ─────────────────────────── ENTRY POINT ─────────────────────────────────────

def main():
    setup_gpio()

    detector = YOLOVehicleDetector(
        model_path        = YOLO_MODEL_PATH,
        conf_threshold    = CONF_THRESHOLD,
        input_size        = YOLO_INPUT_SIZE,
        vehicle_class_ids = VEHICLE_CLASS_IDS,
    )

    # Camera cycling + detection runs in a background daemon thread
    detection_thread = Thread(
        target = camera_and_detection_loop,
        args   = (detector,),
        daemon = True,
    )
    detection_thread.start()

    # HTTP dashboard runs on the main thread (Ctrl-C to quit cleanly)
    try:
        srv = StreamingServer(("", HTTP_PORT), StreamingHandler)
        print(f"[HTTP] Dashboard → http://<pi-ip>:{HTTP_PORT}/")
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[EXIT] Shutting down…")
    finally:
        gp.cleanup()


if __name__ == "__main__":
    main()