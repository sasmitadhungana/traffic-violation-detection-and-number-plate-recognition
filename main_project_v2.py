"""
Red Light Violation Detection & Number Plate Recognition System
================================================================

What this does, in plain terms:

  A camera watches an intersection. This script watches the video back
  and, for each lane, keeps an eye on two things at once: where the
  vehicles are, and what that lane's own traffic light is showing. The
  moment a vehicle crosses its lane's stop line while that lane's light
  is confirmed red, it's logged as a violation - a screenshot of the
  vehicle is saved, its plate is read via OCR, and everything is written
  to a CSV report you can hand someone as evidence.

Why lanes matter: at a real intersection, a left-turn arrow and the
straight-ahead light next to it can show different colors at the same
time. So instead of tracking one signal for the whole frame, each lane
gets its own stop line and its own light to watch (see the LANES setting
below) - a single-lane setup still works fine, it just uses one lane
automatically.

What you get out of a run:
  - output/tracking_output.mp4      - the full video, annotated
  - output/violation_report.csv     - one row per violation
  - output/violations/              - screenshots + plate crops + a
                                       combined "citation" image per violation
  - output/logs.txt                 - a copy of everything printed

Built-in safety nets so a long video doesn't ruin your day:
  - violations are written to the CSV as they happen, not just at the end
  - old per-vehicle bookkeeping gets cleaned up periodically so memory
    doesn't grow forever over a multi-hour recording
  - a dropped/corrupted frame gets retried instead of ending the run early
"""

import os
import sys
import csv
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths - UPDATE THESE TO MATCH YOUR FILES
TRAFFIC_MODEL = os.path.join(BASE_DIR, "models", "traffic_best.pt")
VEHICLE_MODEL = os.path.join(BASE_DIR, "models", "vehicle_best.pt")
PLATE_MODEL = os.path.join(BASE_DIR, "models", "plate_best.pt")

# Video file - UPDATE THIS TO MATCH YOUR VIDEO
VIDEO_FILE = os.path.join(BASE_DIR, "videos", "testvideo", "test7min.mp4")

# Output folders
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")
EVIDENCE_FOLDER = os.path.join(OUTPUT_FOLDER, "violations")
REPORTS_FOLDER = os.path.join(OUTPUT_FOLDER, "reports")

for folder in [OUTPUT_FOLDER, EVIDENCE_FOLDER, REPORTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")

# ======================================================================
# DETECTION SETTINGS
# ======================================================================

# Two separate confidence levels for the traffic light model:
#  - a low "candidate" floor so predict() doesn't throw away small/far lights
#  - a higher floor used only to decide whether a state is trustworthy
TRAFFIC_CANDIDATE_CONF = 0.10
TRAFFIC_ACCEPT_CONF = 0.20

VEHICLE_CONF = 0.30
PLATE_CONF = 0.30

# Run YOLO inference on a smaller, fixed image
# size instead of full source resolution - this is usually the single
# biggest speed lever on CPU. 1920x1080 CCTV frames are far larger than
# YOLO needs; 640 keeps small/far objects (traffic lights) recognizable
# while cutting inference cost substantially. Raise toward 960-1280 only
# if you need to recover accuracy on very small/far plates or lights.
DETECTION_IMGSZ = 640

# The traffic light itself does not need to be re-detected every
# single frame - signals change at most every few seconds. Running that
# model only every N frames (and reusing the last known lights/state on
# the skipped frames) saves a meaningful chunk of total inference time
# without hurting the red-confirmation logic, which already averages over
# several frames anyway.
TRAFFIC_LIGHT_SKIP_FRAMES = 2

# Set to "cuda" if you have a working GPU + matching PyTorch/CUDA
# build (biggest possible speed win - often 5-10x). Leave "cpu" otherwise.
INFERENCE_DEVICE = "cpu"
# Half-precision (FP16) inference - only takes effect on CUDA;
# ignored automatically on CPU. Extra free speedup when DEVICE="cuda".
HALF_PRECISION = False

RED_CONFIRM_FRAMES = 5      # consecutive-ish red detections needed to "confirm" red
STATE_HISTORY_LEN = 10
STATE_VOTE_MIN = 3          # how many of the recent frames must agree

# ---- Long-video robustness settings ----

# Turn OFF the live cv2.imshow preview window for a real speed boost on
# long/batch/headless runs - GUI rendering + waitKey overhead is not free.
# the output video and violations are still produced identically; you
# just don't get a live preview. Press Ctrl+C in the terminal to stop
# early instead of pressing 'q' (no window = no key capture).
SHOW_WINDOW = True

# How often (in frames) to prune bookkeeping dictionaries (vehicle
# history/last-seen/first-seen-side) for vehicle IDs that have not been
# seen in a while. Without this, a multi-hour video with thousands of
# distinct vehicle IDs would grow these dicts forever.
MEMORY_CLEANUP_EVERY_N_FRAMES = 200
VEHICLE_STALE_AFTER_FRAMES = 300   # consider an ID "gone" after this many frames unseen

# Write each violation to the CSV report immediately as it happens (not
# only once at the very end), so a crash/power-loss partway through a
# long video does not lose already-detected violations.
INCREMENTAL_CSV_WRITE = True

# If a frame read fails, retry this many times before concluding the
# video has actually ended - protects against a single corrupted/dropped
# frame in a long recording being mistaken for end-of-video.
FRAME_READ_RETRIES = 5

# --------------------------------------------------------------------
# stop line & signal setup
# --------------------------------------------------------------------
# at a real intersection, different lanes often answer to different
# lights - a left-turn arrow can be red while the straight-ahead light
# next to it is green. So instead of tracking just "the" signal and "the"
# stop line, we describe each lane separately below, and each one gets
# its own stop line and its own signal to watch.
#
# if you only have one lane/one light (the simple case), just leave
# LANES empty - the values right below (STOP_LINE_Y etc.) are used to
# build a single lane automatically, and everything behaves exactly like
# before.

# Fallback / simple single-lane settings (used only when LANES is empty):
STOP_LINE_Y = 565  # moved up 15px from 580 so it lines up with the crosswalk's front edge
STOP_LINE_P1 = None   # e.g. (0, 560)     - set both P1 and P2 for an angled line instead
STOP_LINE_P2 = None   # e.g. (1920, 610)

# Real per-lane setup. Add one dict per lane/route that has its own
# signal. Leave this list empty to use the single-lane fallback above.
#
#   name        - shown on screen and in the report, so use something
#                 readable like "Lane 1 - Straight" or "Left turn lane"
#   x_range     - (x_min, x_max): a vehicle belongs to this lane if its
#                 center falls inside this horizontal slice of the frame
#   stop_line_y - this lane's own stop line (flat horizontal)
#   p1, p2      - OR give two points instead of stop_line_y for an angled
#                 line that matches this lane's real road perspective
#   light_roi   - (x1, y1, x2, y2) box drawn around the ONE signal that
#                 controls this lane. Leave as None and the lane will
#                 just pick whichever signal it can see best - only do
#                 this if there's truly one shared light for everyone.
#
# Starting point for a typical wide approach with 3 signal heads over
# the far lanes plus a nearer lane/route on the right that sits closer
# to the camera (and so needs its own, lower, stop line - perspective
# means "further away" and "closer" can't share one flat line). These
# numbers are a reasonable starting guess, not exact - use
# CALIBRATE_STOP_LINE below (or the '['/']'/'n' keys while the video is
# playing) to nudge each one into place for your actual footage:
LANES = [
    {"name": "Lane 1 - Left turn",  "x_range": (0, 640),
     "stop_line_y": 565, "light_roi": (900, 260, 1020, 420)},
    {"name": "Lane 2 - Straight",   "x_range": (640, 1280),
     "stop_line_y": 565, "light_roi": (1250, 260, 1360, 420)},
    {"name": "Lane 3 - Right turn", "x_range": (1280, 1750),
     "stop_line_y": 565, "light_roi": (1550, 260, 1660, 420)},
    {"name": "Lane 4 - Near side",  "x_range": (1750, 1920),
     "stop_line_y": 640, "light_roi": None},
]

# Instead of a razor-thin line, count a vehicle as "crossed" once it's
# this many extra pixels past the line - gives a small margin instead of
# an all-or-nothing single pixel row.
STOP_LINE_BAND_PX = 0

# How far (in px) a vehicle's box height/width must be to count (filters noise)
MIN_VEHICLE_SIZE = 30

# Ignore vehicles that barely move over a long stretch of the video -
# these are almost always parked cars or roadside vehicles the model
# mistakes for traffic, not anything approaching the light.
PARKED_WINDOW_FRAMES = 40
PARKED_MOVEMENT_PX = 8

# A vehicle only counts as "crossing" if it's actually moving forward
# (toward the camera) by at least this many pixels a frame - this weeds
# out tracker jitter and anything rolling backward.
MIN_FORWARD_SPEED_PX = 3

# Test mode caps processing at MAX_TEST_FRAMES - handy for a quick check
# on a short clip, but it will silently cut off a long video partway
# through. Defaulted to False so a full-length video actually gets
# processed end to end; flip to True only when you want a fast preview
# run on the first chunk of a video.
TEST_MODE = False
MAX_TEST_FRAMES = 500

# Optional manual ROI override for the main traffic light :
# set to (x1, y1, x2, y2) in pixel coordinates to force the system to only
# ever consider signals inside this box (e.g. the light directly over your
# lane). Leave as None to keep automatic largest-box selection.
TRAFFIC_LIGHT_ROI = None  # e.g. (1250, 260, 1360, 420)

# How many recent frames of each vehicle (image + box) to keep on hand so
# that, if a violation fires, OCR can be retried across several frames of
# that same vehicle instead of only the exact crossing frame - this is
# what makes plate reads noticeably more reliable.
VEHICLE_FRAME_BUFFER_LEN = 8

# Set True to run PaddleOCR on GPU (much faster, and
# sometimes more accurate on higher-res crops). Requires paddlepaddle-gpu
# installed with a working CUDA setup matching it. Falls back to CPU
# automatically if GPU init fails, so it's always safe to try True first.
OCR_USE_GPU = False

# What evidence to save for each violation:
#   "plate_only" - just the number plate screenshot (plate_<id>.jpg) -
#                  the smallest, fastest option, good for long videos
#                  where you mainly need "who ran the light and what was
#                  their plate", not a full photo album per violation.
#   "full"       - plate screenshot + vehicle screenshot + full scene
#                  frame + a combined "citation" image with all of them
#                  together. More useful for a demo/report, but more
#                  disk writes per violation.
EVIDENCE_MODE = "plate_only"

# If True, the FIRST frame of the video is
# shown in a window where you click once on the stop line to set
# STOP_LINE_Y automatically, instead of guessing pixel values by hand.
CALIBRATE_STOP_LINE = False

# Ultralytics' built-in tracker
# forgets a lost vehicle after `track_buffer` frames without a match. The
# default (30) can be too short when a vehicle is briefly hidden behind
# another one at a busy intersection, causing its ID to change. Raising
# this keeps the SAME ID through brief occlusions instead of creating a
# new track (which would let a violating vehicle dodge detection).
TRACK_BUFFER_FRAMES = 60
TRACKER_MATCH_THRESH = 0.75

# Mirror every console message into output/logs.txt as well,
# so the final output structure includes a persistent run log.
LOG_TO_FILE = True

# ======================================================================
# LOGGING SETUP
# ======================================================================

class _Tee:
    """Writes to both the real console and a log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


if LOG_TO_FILE:
    _log_path = os.path.join(OUTPUT_FOLDER, "logs.txt")
    _log_file = open(_log_path, "a", encoding="utf-8")
    _log_file.write(f"\n\n===== RUN STARTED {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    sys.stdout = _Tee(sys.__stdout__, _log_file)

# ======================================================================
# OCR SETUP (Optional)
# ======================================================================

# ======================================================================
# OCR SETUP (Optional)
# ======================================================================

try:
    from paddleocr import PaddleOCR
    OCR_AVAILABLE = True
    print("PaddleOCR available")
except ImportError:
    OCR_AVAILABLE = False
    print("PaddleOCR not installed. Plate reading disabled.")
    print("   Install with: pip install paddleocr paddlepaddle")
    print("   (or paddlepaddle-gpu instead of paddlepaddle if you have a working CUDA setup)")


# ======================================================================
# MAIN DETECTOR CLASS
# ======================================================================

class TrafficViolationDetector:
    def __init__(self):
        print("\n" + "=" * 70)
        print("TRAFFIC VIOLATION DETECTION SYSTEM")
        print("=" * 70)

        # Video properties
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None
        self.fps = None

        # Vehicle tracking
        # vehicle_history[id] = list of (center_x, bottom_y) seen each frame
        self.vehicle_history = defaultdict(list)
        self.violated_vehicles = set()
        self.total_violations = 0
        self.violation_records = []

        # Each vehicle keeps a small rolling album of its own recent
        # frames + box. If it turns out to run a red light, we go back
        # through this album and try OCR on all of them - the exact
        # frame it crosses on is often a bit blurry or at an angle, so
        # A frame from a moment before or after usually reads better.
        self.vehicle_frame_buffer = defaultdict(list)
        self.vehicle_last_seen = {}

        # Which side of the stop line a vehicle was on the FIRST time we
        # ever saw it. If a car is first spotted already past the line
        # (it was mid-crossing, or driving away from us), we never blame
        # it for a violation - we only judge vehicles we actually watched
        # approach the line from behind it.
        self.vehicle_first_seen_side = {}

        # Just IDs, never frames or crops, so this can safely be kept for
        # the whole video without using much memory. Used for accurate
        # "vehicles detected" totals even after the heavier per-vehicle
        # bookkeeping above gets cleaned up periodically (see the main
        # loop) to stop long videos from growing memory forever.
        self.all_vehicle_ids_seen = set()

        # Every lane/route gets its own stop line and its own traffic
        # light memory, since real intersections often run a different
        # signal for each lane. See _build_lanes() for how this is set up.
        self.lanes = self._build_lanes()
        self.active_lane_index = 0  # which lane the '['/']' nudge keys currently adjust

        # Raw traffic-light detections from the model, cached so we don't
        # have to re-run that model on every single frame (see
        # TRAFFIC_LIGHT_SKIP_FRAMES) - signals don't change that fast.
        self._cached_traffic_candidates = []

        # Initialize OCR (PaddleOCR)
        self.ocr_reader = None
        if OCR_AVAILABLE:
            self.ocr_reader = self._init_paddle_ocr()

        # Load models
        self._load_models()

        print(f"\nLanes configured: {len(self.lanes)}")
        for lane in self.lanes:
            line_desc = (f"({lane['p1']} -> {lane['p2']})" if lane['p1'] and lane['p2']
                         else f"Y={lane['stop_line_y']}")
            roi_desc = lane['light_roi'] if lane['light_roi'] else "auto (best light in frame)"
            print(f"  - {lane['name']}: stop line {line_desc}, signal ROI: {roi_desc}")
        print(f"Vehicle Confidence: {VEHICLE_CONF}")
        print(f"Test Mode: {'ON' if TEST_MODE else 'OFF'}")
        if TEST_MODE:
            print(f"Max Frames: {MAX_TEST_FRAMES}")
        print("Controls: 'q'=quit, 's'=save frame, ']'/'['=nudge stop line, 'n'=switch active lane")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    def _init_paddle_ocr(self):
        """
        Sets up PaddleOCR, trying GPU first if OCR_USE_GPU is on, and
        always falling back to CPU rather than letting a missing/broken
        CUDA setup crash the whole program.

        PaddleOCR's constructor arguments have changed a bit between
        versions (some accept use_gpu=, newer ones use device= instead,
        and show_log isn't always available) - so this tries a couple of
        different argument combinations before giving up, instead of
        being locked to one specific PaddleOCR version.
        """
        attempts = []
        if OCR_USE_GPU:
            attempts.append(("GPU", dict(use_angle_cls=True, lang='en', use_gpu=True)))
        attempts.append(("CPU", dict(use_angle_cls=True, lang='en', use_gpu=False)))
        attempts.append(("CPU (no use_gpu arg)", dict(use_angle_cls=True, lang='en')))

        for label, kwargs in attempts:
            try:
                reader = PaddleOCR(**kwargs, show_log=False)
                print(f"OCR Reader initialized ({label}, PaddleOCR)")
                return reader
            except TypeError:
                # this PaddleOCR version doesn't like show_log - try without it
                try:
                    reader = PaddleOCR(**kwargs)
                    print(f"OCR Reader initialized ({label}, PaddleOCR)")
                    return reader
                except Exception as e:
                    print(f"PaddleOCR init attempt ({label}) failed: {e}")
            except Exception as e:
                print(f"PaddleOCR init attempt ({label}) failed: {e}")

        print("Could not initialize PaddleOCR - plate reading will be disabled.")
        return None

    # ------------------------------------------------------------------
    def _build_lanes(self):
        """
        Turns the LANES config into the actual tracking state each lane
        needs while the video runs: its own stop line, its own memory of
        red/green history, and its own "is this lane's light red right
        now" answer.

        If the user hasn't set up LANES, we just build one lane that
        spans the whole frame using the plain STOP_LINE_Y / TRAFFIC_LIGHT_ROI
        settings - so a simple single-signal intersection needs zero
        extra configuration and behaves exactly as before.
        """
        configured = LANES if LANES else [{
            "name": "Lane 1",
            "x_range": (0, 10 ** 6),
            "stop_line_y": STOP_LINE_Y,
            "p1": STOP_LINE_P1,
            "p2": STOP_LINE_P2,
            "light_roi": TRAFFIC_LIGHT_ROI,
        }]

        lanes = []
        for cfg in configured:
            lanes.append({
                "name": cfg.get("name", f"Lane {len(lanes) + 1}"),
                "x_range": cfg.get("x_range", (0, 10 ** 6)),
                "stop_line_y": cfg.get("stop_line_y", STOP_LINE_Y),
                "p1": cfg.get("p1"),
                "p2": cfg.get("p2"),
                "light_roi": cfg.get("light_roi"),
                # Everything below is live state, updated every frame
                "state": "UNKNOWN",
                "state_history": [],
                "red_frames": 0,
                "red_confirmed": False,
                "main_box": None,
                "confidence": None,
            })
        return lanes

    # ------------------------------------------------------------------
    def _lane_for_x(self, x):
        """Which lane does this x-position belong to? Falls back to the
        closest lane by center if x doesn't fall neatly inside any of
        them (e.g. a vehicle drifting near a lane boundary)."""
        for lane in self.lanes:
            x_min, x_max = lane["x_range"]
            if x_min <= x <= x_max:
                return lane

        # No exact match - just pick whichever lane's range is closest
        def _distance(lane):
            x_min, x_max = lane["x_range"]
            center = (x_min + x_max) / 2
            return abs(x - center)

        return min(self.lanes, key=_distance)

    # ------------------------------------------------------------------
    def _lane_line_distance(self, lane, x, y):
        """Signed distance from (x, y) to THIS lane's own stop line -
        positive once the point has passed the line. Works whether the
        lane uses a flat stop_line_y or an angled p1/p2 line."""
        if lane["p1"] is not None and lane["p2"] is not None:
            x1, y1 = lane["p1"]
            x2, y2 = lane["p2"]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 or 1.0
            return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length
        return y - lane["stop_line_y"]

    # ------------------------------------------------------------------
    def _load_models(self):
        """Load YOLO models"""
        print("Loading models...")
        print("-" * 50)

        def _list_available(path):
            """Helper for when a model path is wrong, show what
            .pt files actually exist in that folder so the fix is obvious."""
            folder = os.path.dirname(path)
            if os.path.isdir(folder):
                found = [f for f in os.listdir(folder) if f.lower().endswith(".pt")]
                if found:
                    print(f"      Found these .pt files in {folder}:")
                    for f in found:
                        print(f"        - {f}")
                else:
                    print(f"      No .pt files found in {folder}")
            else:
                print(f"      Folder does not exist: {folder}")

        def _load(path, label):
            if os.path.exists(path):
                try:
                    m = YOLO(path)
                    print(f"  {label}: LOADED")
                    return m
                except Exception as e:
                    print(f"  {label}: FAILED - {e}")
                    return None
            else:
                print(f"  {label}: NOT FOUND at {path}")
                _list_available(path)
                return None

        self.traffic_model = _load(TRAFFIC_MODEL, "Traffic Light Model")
        self.vehicle_model = _load(VEHICLE_MODEL, "Vehicle Model")
        self.plate_model = _load(PLATE_MODEL, "License Plate Model")

        print("-" * 50)

        if self.traffic_model is not None:
            print(f"  Traffic model classes: {self.traffic_model.names}")

        if self.traffic_model is None and self.vehicle_model is None:
            print("\nERROR: No models loaded! Please check model paths.")
            sys.exit(1)

        # Write a custom ByteTrack config with a longer track
        # buffer so a vehicle briefly hidden behind another one keeps its
        # SAME id instead of being re-assigned a new one when it re-appears.
        self.tracker_config_path = self._write_tracker_config()

    # ------------------------------------------------------------------
    def _write_tracker_config(self):
        """generate a tuned bytetrack.yaml next to this script so
        ultralytics' tracker holds onto lost vehicles longer (occlusion
        robustness) and matches slightly more leniently between frames."""
        config_path = os.path.join(BASE_DIR, "custom_bytetrack.yaml")
        content = (
            "tracker_type: bytetrack\n"
            "track_high_thresh: 0.5\n"
            "track_low_thresh: 0.1\n"
            "new_track_thresh: 0.6\n"
            f"track_buffer: {TRACK_BUFFER_FRAMES}\n"
            f"match_thresh: {TRACKER_MATCH_THRESH}\n"
            "fuse_score: True\n"
        )
        try:
            with open(config_path, "w") as f:
                f.write(content)
            print(f"  Tracker config: {config_path} (track_buffer={TRACK_BUFFER_FRAMES})")
            return config_path
        except Exception as e:
            print(f"  Could not write custom tracker config ({e}), using default tracker")
            return "bytetrack.yaml"

    # ------------------------------------------------------------------
    def _detect_traffic_light(self, frame):
        """
        Finds every traffic light in the frame, then lets each LANE work
        out its own signal state from that same list - so a left-turn
        lane can be reading red while the straight-ahead lane next to it
        is reading green, which is exactly how real intersections work.

        We only run the actual detection model once per frame (or once
        every TRAFFIC_LIGHT_SKIP_FRAMES frames - signals don't change
        fast enough to need checking every single frame). Each lane then
        just looks at whichever of those detections fall inside its own
        light_roi and decides its own state from those.
        """
        if self.traffic_model is None:
            return []

        frame_center_x = self.frame_width / 2 if self.frame_width else 0

        if TRAFFIC_LIGHT_SKIP_FRAMES > 1 and self.frame_count % TRAFFIC_LIGHT_SKIP_FRAMES != 0:
            candidates = self._cached_traffic_candidates
        else:
            try:
                results = self.traffic_model.predict(
                    frame,
                    conf=TRAFFIC_CANDIDATE_CONF,
                    imgsz=DETECTION_IMGSZ,
                    device=INFERENCE_DEVICE,
                    half=HALF_PRECISION,
                    verbose=False
                )
            except Exception:
                results = None

            candidates = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = self.traffic_model.names[class_id].lower()

                    if "red" in label:
                        norm_label, color = "red", (0, 0, 255)
                    elif "green" in label:
                        norm_label, color = "green", (0, 255, 0)
                    elif "yellow" in label or "amber" in label:
                        norm_label, color = "yellow", (0, 255, 255)
                    else:
                        continue

                    if confidence < TRAFFIC_ACCEPT_CONF:
                        continue

                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    candidates.append({
                        'box': (x1, y1, x2, y2),
                        'color': color,
                        'label': norm_label,
                        'confidence': confidence,
                        'area': area,
                        'center_x': center_x,
                        'center_y': center_y,
                    })

            self._cached_traffic_candidates = candidates

        # Now let each lane pick its own signal out of the same detections.
        for lane in self.lanes:
            roi = lane["light_roi"]
            if roi:
                rx1, ry1, rx2, ry2 = roi
                pool = [
                    c for c in candidates
                    if rx1 <= c['center_x'] <= rx2 and ry1 <= c['center_y'] <= ry2
                ]
                # If nothing was actually seen inside this lane's ROI this
                # frame, don't just grab an unrelated light from elsewhere
            else:
                pool = candidates

            current_state = "UNKNOWN"
            lane["main_box"] = None
            if pool:
                main = sorted(
                    pool,
                    key=lambda c: (-c['confidence'], -c['area'],
                                   abs(c['center_x'] - frame_center_x))
                )[0]
                current_state = main['label'].upper()
                lane["main_box"] = main['box']
                lane["confidence"] = main['confidence']
                if current_state == "RED":
                    lane["red_frames"] += 1

            lane["state_history"].append(current_state)
            if len(lane["state_history"]) > STATE_HISTORY_LEN:
                lane["state_history"].pop(0)

            # Confirm the state by majority vote over recent frames, so a
            # single flickering misdetection doesn't flip the whole lane.
            if len(lane["state_history"]) >= 5:
                votes = defaultdict(int)
                for s in lane["state_history"]:
                    if s != "UNKNOWN":
                        votes[s] += 1
                if votes:
                    most_common, count = max(votes.items(), key=lambda x: x[1])
                    if count >= STATE_VOTE_MIN:
                        lane["state"] = most_common
                        lane["red_confirmed"] = (
                            most_common == "RED" and lane["red_frames"] >= RED_CONFIRM_FRAMES
                        )
                        if most_common != "RED":
                            lane["red_frames"] = 0

        return candidates

    # ------------------------------------------------------------------
    def _detect_vehicles(self, frame):
        """Detect and track vehicles"""
        if self.vehicle_model is None:
            return []

        try:
            # Use the tuned tracker config (longer track_buffer)
            # so IDs survive brief occlusions instead of switching.
            # smaller imgsz + device/half for faster inference.
            results = self.vehicle_model.track(
                frame,
                persist=True,
                conf=VEHICLE_CONF,
                imgsz=DETECTION_IMGSZ,
                device=INFERENCE_DEVICE,
                half=HALF_PRECISION,
                tracker=self.tracker_config_path,
                verbose=False
            )
        except Exception:
            # Fall back to the default tracker if the custom config path
            # ever has a problem (e.g. ultralytics version mismatch).
            try:
                results = self.vehicle_model.track(
                    frame, persist=True, conf=VEHICLE_CONF,
                    imgsz=DETECTION_IMGSZ, verbose=False
                )
            except Exception:
                return []

        vehicles = []

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_id = int(box.id[0])
                confidence = float(box.conf[0])

                height = y2 - y1
                width = x2 - x1
                if height < MIN_VEHICLE_SIZE or width < MIN_VEHICLE_SIZE:
                    continue

                vehicles.append({
                    'id': vehicle_id,
                    'box': (x1, y1, x2, y2),
                    'center_x': int((x1 + x2) / 2),
                    'center_y': int((y1 + y2) / 2),
                    'bottom_y': y2,
                    'confidence': confidence
                })

        return vehicles

    # ------------------------------------------------------------------
    def _detect_plate(self, frame, vehicle_box):
        """
        Detect the license plate that belongs to THIS specific vehicle
        : we only ever search inside the padded crop of the
        vehicle's own bounding box, so a plate detected here cannot be
        confused with a different vehicle's plate.
        """
        if self.plate_model is None:
            return None

        x1, y1, x2, y2 = vehicle_box
        padding = 15

        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(self.frame_width, x2 + padding)
        y2p = min(self.frame_height, y2 + padding)

        vehicle_region = frame[y1p:y2p, x1p:x2p]
        if vehicle_region.size == 0:
            return None

        try:
            results = self.plate_model.predict(
                vehicle_region,
                conf=PLATE_CONF,
                device=INFERENCE_DEVICE,
                half=HALF_PRECISION,
                verbose=False
            )
        except Exception:
            return None

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Take the highest-confidence plate found inside this vehicle's crop
            best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
            px1, py1, px2, py2 = map(int, best_box.xyxy[0])
            confidence = float(best_box.conf[0])

            abs_x1, abs_y1 = x1p + px1, y1p + py1
            abs_x2, abs_y2 = x1p + px2, y1p + py2

            plate_image = frame[abs_y1:abs_y2, abs_x1:abs_x2]
            if plate_image.size > 0:
                return {
                    'image': plate_image,
                    'confidence': confidence,
                    'box': (abs_x1, abs_y1, abs_x2, abs_y2)
                }

        return None

    # ------------------------------------------------------------------
    def _ocr_variants(self, gray):
        """
        Plate-recognition quality improvement: generates several different
        preprocessed versions of the same plate crop, since no single
        preprocessing pipeline reads every plate well (lighting, glare,
        and plate condition vary a lot across an intersection). Returns
        a list of (name, image) pairs to try OCR on.
        """
        variants = []

        # 1. Plain upscaled grayscale - sometimes best for small/antialiased text
        h, w = gray.shape
        scale = 300 / w if w > 0 else 1
        new_w = 300
        new_h = max(1, int(h * scale))
        plain = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        variants.append(("plain", plain))

        # 2. CLAHE contrast enhancement - helps faded/low-contrast plates
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        variants.append(("clahe", enhanced))

        # 3. Adaptive threshold - copes with uneven lighting/shadows
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )
        adaptive = cv2.resize(adaptive, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        variants.append(("adaptive", adaptive))

        # 4. Otsu global threshold - works well when lighting is even
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = cv2.resize(otsu, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        variants.append(("otsu", otsu))

        return variants

    # ------------------------------------------------------------------
    def _flatten_paddle_result(self, raw):
        """
        PaddleOCR's exact return shape has shifted a bit across versions
        (nested list of [box, (text, confidence)] pairs in most releases,
        occasionally wrapped differently in others). This just normalizes
        whatever comes back into a plain list of (box, text, confidence)
        tuples so the rest of the code doesn't need to care which version
        is installed.
        """
        lines = []
        if not raw:
            return lines

        first = raw[0] if isinstance(raw, list) else raw
        if not first:
            return lines

        for item in first:
            try:
                box, text_conf = item
                text, conf = text_conf
                lines.append((box, text, float(conf)))
            except Exception:
                continue
        return lines

    # ------------------------------------------------------------------
    def _clean_plate_text(self, text):
        """PaddleOCR doesn't have EasyOCR's allowlist option, so we just
        strip anything that isn't a letter or digit ourselves afterward."""
        return re.sub(r'[^A-Z0-9]', '', text.strip().upper())

    # ------------------------------------------------------------------
    def _ocr_single(self, image):
        """Runs PaddleOCR on one already-preprocessed image and returns
        (text, confidence), combining left-to-right boxes into one string
        (plates are sometimes split into two OCR boxes)."""
        if self.ocr_reader is None:
            return None, 0.0

        # PaddleOCR expects a 3-channel image, our preprocessed variants
        # are often single-channel grayscale/threshold images.
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        try:
            raw = self.ocr_reader.ocr(image, cls=True)
        except TypeError:
            # some PaddleOCR versions no longer accept the cls= argument
            try:
                raw = self.ocr_reader.ocr(image)
            except Exception:
                return None, 0.0
        except Exception:
            return None, 0.0

        lines = self._flatten_paddle_result(raw)
        if not lines:
            return None, 0.0

        lines = sorted(lines, key=lambda l: min(p[0] for p in l[0]))  # left to right
        used = [l for l in lines if l[2] >= 0.25]
        combined = "".join(self._clean_plate_text(l[1]) for l in used)
        avg_conf = sum(l[2] for l in used) / len(used) if used else 0.0

        if 2 <= len(combined) <= 12:
            return combined, avg_conf

        best = max(lines, key=lambda l: l[2])
        text = self._clean_plate_text(best[1])
        if 2 <= len(text) <= 12:
            return text, float(best[2])

        return None, 0.0

    # ------------------------------------------------------------------
    def _read_plate_text(self, plate_image):
        """
        OCR pipeline, upgraded for read quality: tries several
        preprocessing variants (plain, CLAHE, adaptive threshold, Otsu
        threshold) on the SAME plate crop and keeps whichever gives the
        highest-confidence readable text, instead of committing to a
        single fixed pipeline that may not suit every plate/lighting.

        Returns (text, confidence) so callers can also compare reads
        taken from different frames of the same vehicle. A failed/empty
        read returns (None, 0.0).
        """
        if self.ocr_reader is None or plate_image is None or plate_image.size == 0:
            return None, 0.0

        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None, 0.0

        best_text, best_conf = None, -1.0
        for _name, variant_img in self._ocr_variants(gray):
            text, conf = self._ocr_single(variant_img)
            if text and conf > best_conf:
                best_text, best_conf = text, conf

        if best_text is not None:
            return best_text, max(best_conf, 0.0)
        return None, 0.0

    # ------------------------------------------------------------------
    def _best_plate_for_vehicle(self, vehicle_id, fallback_frame, fallback_box):
        """
        Explicit vehicle -> plate -> OCR link (fixes #1, #2, #3):

            Vehicle <id>
                |
                +-- Plate (detected fresh inside THIS vehicle's own box,
                |          on every buffered frame of that same vehicle)
                |
                +-- OCR (best-confidence read across those frames)

        Tries every buffered frame for this vehicle ID (fix for reliability:
        the exact crossing frame is often motion-blurred or angled, so we
        also try a few frames right before/after it) and keeps whichever
        (plate image, text, confidence) combination scored highest.
        Falls back to the single current frame if no buffer exists.
        """
        buffer = self.vehicle_frame_buffer.get(vehicle_id, [])
        candidate_frames = list(buffer) if buffer else []
        candidate_frames.append({'frame': fallback_frame, 'box': fallback_box})

        best_plate_image = None
        best_text = None
        best_score = -1.0

        for entry in candidate_frames:
            plate = self._detect_plate(entry['frame'], entry['box'])
            if not plate:
                continue

            text, ocr_conf = self._read_plate_text(plate['image'])
            # Score combines plate-detector confidence and OCR confidence,
            # and rewards actually getting readable text
            score = plate['confidence'] * 0.4 + ocr_conf * 0.6
            if text:
                score += 0.1

            if score > best_score:
                best_score = score
                best_plate_image = plate['image']
                best_text = text

        return best_plate_image, best_text

    # ------------------------------------------------------------------
    def _update_crossing_state(self, vehicle):
        """
        Compares the vehicle's front-bumper point (bottom-center of its
        box) between the previous frame and this one, rather than just
        looking at a single frame in isolation - that way a vehicle has
        to actually be seen moving across the line, not just happen to
        be detected past it once.

        The vehicle is checked against its OWN lane's stop line (each
        lane can sit at a slightly different spot / angle).

        Also requires the vehicle to be moving forward, toward the
        camera, by a minimum amount - this rules out tracker jitter or a
        vehicle rolling backward from accidentally being counted.

        Returns True exactly on the frame the vehicle goes from being
        before the line to being past it.
        """
        vehicle_id = vehicle['id']
        history = self.vehicle_history[vehicle_id]

        if len(history) < 2:
            return False

        prev_x, prev_y = history[-2]
        curr_x, curr_y = history[-1]

        forward_speed = curr_y - prev_y  # positive = moving down/toward camera
        if forward_speed < MIN_FORWARD_SPEED_PX:
            return False

        lane = self._lane_for_x(curr_x)
        prev_dist = self._lane_line_distance(lane, prev_x, prev_y)
        curr_dist = self._lane_line_distance(lane, curr_x, curr_y)

        return prev_dist < STOP_LINE_BAND_PX <= curr_dist

    # ------------------------------------------------------------------
    def _is_parked_or_irrelevant(self, vehicle_id):
        """
        treat a vehicle as parked/irrelevant (and skip violation
        checks for it) if it has barely moved over the last
        PARKED_WINDOW_FRAMES frames - filters out roadside parked cars
        the model may pick up as "traffic".
        """
        history = self.vehicle_history[vehicle_id]
        if len(history) < PARKED_WINDOW_FRAMES:
            return False

        window = history[-PARKED_WINDOW_FRAMES:]
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        displacement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return displacement < PARKED_MOVEMENT_PX

    # ------------------------------------------------------------------
    def _save_violation_citation(self, output_frame, vehicle_crop, plate_image,
                                  vehicle_id, plate_text, video_timestamp, lane_name="Lane 1"):
        """
        Builds one single "here's the proof" image for a violation - the
        kind of thing you'd actually hand someone as evidence: the full
        scene up top (with the stop line, the red light, and the vehicle
        box already drawn on it), and a close-up screenshot of the
        vehicle and its plate underneath, plus a header with all the
        details. Saved as Violation_001.jpg, Violation_002.jpg, etc.
        """
        try:
            scene = output_frame.copy()
            h, w = scene.shape[:2]

            banner_h = 70
            inset_h = 130  # a bit larger so the vehicle screenshot reads clearly
            citation = np.zeros((h + banner_h + inset_h + 20, w, 3), dtype=np.uint8)
            citation[:] = (20, 20, 20)
            citation[banner_h:banner_h + h, 0:w] = scene

            header = (f"VIOLATION #{self.total_violations:03d}   "
                      f"Vehicle ID: {vehicle_id}   "
                      f"Plate: {plate_text or 'UNREADABLE'}   "
                      f"Lane: {lane_name}   "
                      f"Time: {video_timestamp}   Signal: RED")
            cv2.putText(citation, header, (15, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            # Inset strip: the violating vehicle's own screenshot first
            # (this is the main piece of evidence), then the plate crop.
            strip_y = banner_h + h + 10
            x_cursor = 15

            def _paste_inset(img, label):
                nonlocal x_cursor
                if img is None or img.size == 0:
                    return
                ih, iw = img.shape[:2]
                scale = inset_h / ih
                resized = cv2.resize(img, (max(1, int(iw * scale)), inset_h))
                rw = resized.shape[1]
                if x_cursor + rw < w:
                    citation[strip_y:strip_y + inset_h, x_cursor:x_cursor + rw] = resized
                    cv2.putText(citation, label, (x_cursor, strip_y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    x_cursor += rw + 20

            _paste_inset(vehicle_crop, "VIOLATING VEHICLE")
            _paste_inset(plate_image, "PLATE")

            filename = f"Violation_{self.total_violations:03d}.jpg"
            path = os.path.join(EVIDENCE_FOLDER, filename)
            cv2.imwrite(path, citation)
            return path
        except Exception as e:
            print(f"Could not build citation image: {e}")
            return None

    # ------------------------------------------------------------------
    def _check_violation(self, vehicle, frame, output_frame):
        """
        This is the heart of the whole system. A violation is only ever
        recorded if EVERY one of these is true:
            1. This vehicle's OWN lane has a confirmed red light
            2. We watched this vehicle approach from before the line -
               we never blame a car we only ever saw already past it
            3. It's not just sitting there (parked/stationary) - it
               actually crosses the line
            4. It's genuinely moving forward, not tracker jitter
            5. It hasn't already been counted once before
        """
        vehicle_id = vehicle['id']

        if vehicle_id in self.violated_vehicles:
            return False

        # Never blame a vehicle we first detected already past the line -
        # we can't vouch for what it did before it entered view.
        if self.vehicle_first_seen_side.get(vehicle_id) != "BEFORE":
            return False

        if self._is_parked_or_irrelevant(vehicle_id):
            return False

        crossed_now = self._update_crossing_state(vehicle)
        if not crossed_now:
            return False

        # Check THIS vehicle's own lane - not some other lane's light.
        lane = self._lane_for_x(vehicle['center_x'])
        if not lane["red_confirmed"]:
            return False

        # ---- VIOLATION DETECTED ----
        self.violated_vehicles.add(vehicle_id)
        self.total_violations += 1

        # Vehicle -> plate -> OCR, tried across every frame we've kept of
        # this vehicle - much more reliable than reading only the exact
        # frame it happened to cross on.
        best_plate_image, plate_text = self._best_plate_for_vehicle(
            vehicle_id, frame, vehicle['box']
        )

        # The plate screenshot is always saved - this is the one piece of
        # evidence every mode keeps, since it's the whole point of the
        # "number plate recognition" part of the system.
        plate_file = None
        if best_plate_image is not None and best_plate_image.size > 0:
            plate_file = os.path.join(EVIDENCE_FOLDER, f"plate_{vehicle_id}.jpg")
            cv2.imwrite(plate_file, best_plate_image)

        video_seconds = self.frame_count / self.fps if self.fps else 0
        video_timestamp = str(timedelta(seconds=int(video_seconds)))

        frame_file = None
        vehicle_file = None
        violation_file = None

        if EVIDENCE_MODE == "full":
            # The full scene at the moment of the violation
            frame_file = os.path.join(EVIDENCE_FOLDER, f"frame_{self.frame_count}.jpg")
            cv2.imwrite(frame_file, output_frame)

            # A screenshot of just the violating vehicle
            x1, y1, x2, y2 = vehicle['box']
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size > 0:
                vehicle_file = os.path.join(EVIDENCE_FOLDER, f"vehicle_{vehicle_id}.jpg")
                cv2.imwrite(vehicle_file, vehicle_crop)

            # One combined "citation" image - the full scene plus close-up
            # crops of the vehicle and plate, with a header banner.
            violation_file = self._save_violation_citation(
                output_frame, vehicle_crop if vehicle_crop.size > 0 else None,
                best_plate_image, vehicle_id, plate_text, video_timestamp, lane["name"]
            )

        record = {
            'id': len(self.violation_records) + 1,
            'vehicle_id': vehicle_id,
            'lane': lane["name"],
            'frame': self.frame_count,
            'video_time': video_timestamp,
            'wall_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'plate_text': plate_text or "UNREADABLE",
            'state': lane["state"],
            'evidence_frame': frame_file,
            'evidence_vehicle': vehicle_file,
            'evidence_plate': plate_file,
            'evidence_citation': violation_file
        }
        self.violation_records.append(record)

        # Save this violation to the CSV right away, not just at the end -
        # if the program is stopped partway through a long video, nothing
        # already found is lost.
        self._append_violation_to_csv(record)

        print("\n" + "=" * 70)
        print(f"VIOLATION #{self.total_violations} DETECTED!")
        print("=" * 70)
        print(f"  Vehicle ID    : {vehicle_id}")
        print(f"  Lane          : {lane['name']}")
        print(f"  Frame         : {self.frame_count}")
        print(f"  Video Time    : {video_timestamp}")
        print(f"  Traffic State : {lane['state']}")
        print(f"  Plate Number  : {record['plate_text']}")
        print("-" * 70)
        print("  Evidence saved:")
        if plate_file:
            print(f"    Plate    : {os.path.basename(plate_file)}  <-- number plate screenshot")
        if frame_file:
            print(f"    Frame    : {os.path.basename(frame_file)}")
        if vehicle_file:
            print(f"    Vehicle  : {os.path.basename(vehicle_file)}")
        if violation_file:
            print(f"    Citation : {os.path.basename(violation_file)}")
        print("=" * 70 + "\n")

        return True

    # ------------------------------------------------------------------
    def _draw_annotations(self, frame, lights, vehicles):
        """Draws everything onto the frame: each lane's stop line and
        signal status, the vehicle boxes, and the on-screen dashboard."""
        output = frame.copy()

        def _status_for(state, confirmed_red):
            if confirmed_red or state == "RED":
                return "RED", (0, 0, 255), (0, 0, 100)
            if state == "GREEN":
                return "GREEN", (0, 255, 0), (0, 80, 0)
            if state == "YELLOW":
                return "YELLOW", (0, 255, 255), (80, 80, 0)
            return "NO SIGNAL", (255, 255, 0), (80, 80, 0)

        # 1. Each lane draws its own stop line - ONLY across that lane's
        # own width (x_range), not the full frame. Otherwise, with more
        # than one lane configured, every line would be drawn on top of
        # the others across the whole frame and you'd never be able to
        # tell them apart.
        for lane in self.lanes:
            status_text, status_color, _ = _status_for(lane["state"], lane["red_confirmed"])
            line_color = status_color if status_text == "RED" else (0, 255, 255)

            if lane["p1"] is not None and lane["p2"] is not None:
                p1, p2 = lane["p1"], lane["p2"]
                cv2.line(output, p1, p2, line_color, 4)
                label_x, label_y = p1[0], p1[1]
            else:
                y = lane["stop_line_y"]
                x_min, x_max = lane["x_range"]
                x_start = max(0, int(x_min))
                x_end = min(self.frame_width, int(x_max))
                cv2.line(output, (x_start, y), (x_end, y), line_color, 4)
                for x in range(x_start + 20, x_end, 50):
                    cv2.line(output, (x, y - 4), (x + 25, y - 4), line_color, 2)
                label_x, label_y = x_start + 10, y

            label = f"{lane['name']} STOP LINE"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output, (label_x - 5, label_y - text_size[1] - 12),
                          (label_x + text_size[0] + 10, label_y + 8), (0, 0, 0), -1)
            cv2.putText(output, label, (label_x, label_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

        # 2. Each lane's own signal box, panel, and confidence.
        panel_h = 34 + 26 * len(self.lanes) + 10
        cv2.rectangle(output, (10, 10), (480, 10 + panel_h), (0, 0, 0), -1)
        cv2.putText(output, "SIGNAL STATUS BY LANE", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        row_y = 58
        for lane in self.lanes:
            status_text, status_color, _ = _status_for(lane["state"], lane["red_confirmed"])
            conf_txt = f"{lane['confidence']:.2f}" if lane["confidence"] else "-"
            cv2.putText(output, f"{lane['name']}: {status_text}  (conf {conf_txt}, "
                                 f"red frames {lane['red_frames']})",
                        (20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
            row_y += 26

        video_seconds = self.frame_count / self.fps if self.fps else 0
        video_timestamp = str(timedelta(seconds=int(video_seconds)))

        # 3. Draw every detected light, highlighting whichever one each
        # lane picked as its own "main" signal.
        main_boxes = {lane["main_box"] for lane in self.lanes if lane["main_box"]}
        for light in lights:
            x1, y1, x2, y2 = light['box']
            is_main = light['box'] in main_boxes
            thickness = 3 if is_main else 1
            cv2.rectangle(output, (x1, y1), (x2, y2), light['color'], thickness)
            label_txt = light['label'].upper() + (" [USED]" if is_main else "")
            cv2.putText(output, label_txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, light['color'], 2)

        # 4. Draw each vehicle, colored red once it's flagged as a violation.
        for vehicle in vehicles:
            vehicle_id = vehicle['id']
            x1, y1, x2, y2 = vehicle['box']
            has_violation = vehicle_id in self.violated_vehicles

            plate_text = None
            if has_violation:
                for record in self.violation_records:
                    if record['vehicle_id'] == vehicle_id:
                        plate_text = record['plate_text']
                        break

            if has_violation:
                color = (0, 0, 255)
                thickness = 3
                cv2.putText(output, "VIOLATION!", (x1, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                color = (0, 255, 0)
                thickness = 2

            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(output, (vehicle['center_x'], vehicle['bottom_y']), 5, (255, 0, 255), -1)

            # Two-line "ID:34 / Plate: BA2CHA5678" label - much easier to
            # read in a demo than squeezing both onto one line.
            if plate_text:
                cv2.rectangle(output, (x1, y1 - 46), (x1 + 230, y1), (0, 0, 0), -1)
                cv2.putText(output, f"ID:{vehicle_id}", (x1 + 5, y1 - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(output, f"Plate: {plate_text}", (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            else:
                cv2.rectangle(output, (x1, y1 - 28), (x1 + 150, y1), (0, 0, 0), -1)
                cv2.putText(output, f"ID:{vehicle_id}", (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 5. General stats panel (frame count, vehicle count, FPS, etc.)
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        proc_fps = self.frame_count / elapsed if elapsed > 0 else 0

        panel_top = 10 + panel_h + 10
        cv2.rectangle(output, (10, panel_top), (560, panel_top + 155), (0, 0, 0), -1)

        y = panel_top + 23
        cv2.putText(output, f"Frame: {self.frame_count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Vehicles: {len(vehicles)}", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Violations: {self.total_violations}", (360, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if self.total_violations > 0 else (255, 255, 255), 1)

        y += 27
        cv2.putText(output, f"FPS: {proc_fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Time: {video_timestamp}", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Total Tracked: {len(self.all_vehicle_ids_seen)}", (360, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        y += 27
        if self.violation_records:
            last = self.violation_records[-1]
            last_txt = (f"Last Violation: ID {last['vehicle_id']}  "
                        f"{last['plate_text']}  ({last.get('lane', 'Lane 1')})")
        else:
            last_txt = "Last Violation: none yet"
        cv2.putText(output, last_txt, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        cv2.putText(output, "q=quit  s=save  [/]=move stop line  n=switch lane",
                    (10, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output

    # ------------------------------------------------------------------
    def _csv_header(self):
        return [
            "Vehicle ID", "Lane", "Plate", "Frame", "Time", "Signal", "Status",
            "Violation ID", "Wall Time",
            "Evidence Frame", "Evidence Vehicle", "Evidence Plate", "Evidence Citation"
        ]

    def _csv_row(self, r):
        return [
            r['vehicle_id'],
            r.get('lane', 'Lane 1'),
            r['plate_text'],
            r['frame'],
            r['video_time'],
            r['state'],
            "Red Light Violation",
            r['id'],
            r['wall_time'],
            os.path.basename(r['evidence_frame']) if r['evidence_frame'] else "",
            os.path.basename(r['evidence_vehicle']) if r['evidence_vehicle'] else "",
            os.path.basename(r['evidence_plate']) if r['evidence_plate'] else "",
            os.path.basename(r['evidence_citation']) if r.get('evidence_citation') else ""
        ]

    # ------------------------------------------------------------------
    def _append_violation_to_csv(self, record):
        """
        Long-video robustness: append this ONE violation to
        violation_report.csv immediately, instead of waiting until the
        whole video finishes to write anything. If the process ever
        crashes, is killed, or the machine loses power partway through a
        multi-hour video, every violation detected up to that point is
        still safely on disk.
        """
        if not INCREMENTAL_CSV_WRITE:
            return
        report_file = os.path.join(OUTPUT_FOLDER, "violation_report.csv")
        file_is_new = not os.path.exists(report_file)
        try:
            with open(report_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if file_is_new:
                    writer.writerow(self._csv_header())
                writer.writerow(self._csv_row(record))
        except Exception as e:
            print(f"Warning: could not append violation to CSV: {e}")

    # ------------------------------------------------------------------
    def _save_report(self):
        """Generate/overwrite the full CSV report .
        Used as the final, complete rewrite at the end of the run - the
        incremental version above already protects against data loss
        mid-run, this just guarantees a clean, fully-ordered final file."""
        # The primary report lives
        # directly in output/ as violation_report.csv, matching the
        # requested final folder layout. A timestamped copy is also kept
        # in output/reports/ so repeated runs don't overwrite history.
        report_file = os.path.join(OUTPUT_FOLDER, "violation_report.csv")
        timestamped_file = os.path.join(
            REPORTS_FOLDER,
            f"violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if not self.violation_records:
            print("\nNo violations detected - writing empty report with headers only")

        for path in (report_file, timestamped_file):
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self._csv_header())
                for r in self.violation_records:
                    writer.writerow(self._csv_row(r))

        print(f"\nReport saved: {report_file}")
        print(f"Timestamped copy: {timestamped_file}")
        return report_file

    # ------------------------------------------------------------------
    def _calibrate_stop_line(self, frame, lane_index=0):
        """
        Shows the first frame in a window and lets you click once on
        where a lane's stop line should be - much easier than guessing
        pixel coordinates by hand. Press any key to confirm, or just
        press a key without clicking to keep the current position.
        """
        lane = self.lanes[lane_index]
        picked = {'y': lane["stop_line_y"]}
        window = f"Click the stop line for {lane['name']}, then press any key"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                picked['y'] = y

        preview = frame.copy()
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_click)

        while True:
            display = preview.copy()
            cv2.line(display, (0, picked['y']), (self.frame_width, picked['y']), (0, 255, 255), 3)
            cv2.putText(display, f"{lane['name']}: Stop Line Y = {picked['y']}  "
                                  f"(click to move, any key to confirm)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(window, display)
            if cv2.waitKey(30) != -1:
                break

        cv2.destroyWindow(window)
        print(f"{lane['name']} stop line calibrated to Y = {picked['y']}")
        return picked['y']

    # ------------------------------------------------------------------
    def process_video(self):
        """Main processing loop"""

        if not os.path.exists(VIDEO_FILE):
            print(f"ERROR: Video not found: {VIDEO_FILE}")
            print("Please update VIDEO_FILE path in the configuration.")
            video_folder = os.path.dirname(VIDEO_FILE)
            if os.path.isdir(video_folder):
                found = [f for f in os.listdir(video_folder)
                         if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
                if found:
                    print(f"  Found these video files in {video_folder}:")
                    for f in found:
                        print(f"    - {f}")
                else:
                    print(f"  No video files found in {video_folder}")
            else:
                print(f"  Folder does not exist: {video_folder}")
            return

        cap = cv2.VideoCapture(VIDEO_FILE)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {VIDEO_FILE}")
            return

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Optional one-click calibration on the first frame instead of
        # guessing stop-line pixel coordinates by hand. Calibrates each
        # lane one at a time if you have more than one.
        if CALIBRATE_STOP_LINE:
            ret, calib_frame = cap.read()
            if ret:
                for i, lane in enumerate(self.lanes):
                    lane["stop_line_y"] = self._calibrate_stop_line(calib_frame, lane_index=i)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start after calibrating

        print("\n" + "=" * 60)
        print("STARTING VIDEO PROCESSING")
        print("=" * 60)
        print(f"Video: {os.path.basename(VIDEO_FILE)}")
        print(f"Size: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps:.1f}")
        for lane in self.lanes:
            print(f"{lane['name']} stop line: Y={lane['stop_line_y']}")
        print("=" * 60 + "\n")

        output_video = os.path.join(OUTPUT_FOLDER, "tracking_output.mp4")
        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height)
        )

        self.start_time = time.time()
        read_failures = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                # Long-video robustness: a single dropped/corrupted frame
                # in a multi-hour recording shouldn't be mistaken for
                # end-of-video. Retry a few times before giving up.
                read_failures += 1
                if read_failures <= FRAME_READ_RETRIES:
                    print(f"Warning: frame read failed (attempt {read_failures}/"
                          f"{FRAME_READ_RETRIES}), retrying...")
                    continue
                print("\nEnd of video reached (or unrecoverable read failure)")
                break

            read_failures = 0  # reset on any successful read
            self.frame_count += 1

            # 1. Traffic light state (main-light selection fix)
            lights = self._detect_traffic_light(frame)

            # 2. Vehicles
            vehicles = self._detect_vehicles(frame)

            # 3. Update per-vehicle position history
            for v in vehicles:
                vid = v['id']

                # Remember which side of ITS lane's stop line this vehicle
                # was on the first time we ever saw it tracked - used
                # later to make sure we only ever judge vehicles we
                # actually watched approach the line.
                if vid not in self.vehicle_first_seen_side:
                    lane = self._lane_for_x(v['center_x'])
                    dist = self._lane_line_distance(lane, v['center_x'], v['bottom_y'])
                    self.vehicle_first_seen_side[vid] = "BEFORE" if dist < STOP_LINE_BAND_PX else "AFTER"

                self.vehicle_history[vid].append((v['center_x'], v['bottom_y']))
                if len(self.vehicle_history[vid]) > 20:
                    self.vehicle_history[vid].pop(0)

                # Keep a short rolling buffer of this vehicle's own raw
                # frame + box so a later violation can retry plate OCR
                # across several frames of the SAME vehicle .
                buf = self.vehicle_frame_buffer[vid]
                buf.append({'frame': frame.copy(), 'box': v['box']})
                if len(buf) > VEHICLE_FRAME_BUFFER_LEN:
                    buf.pop(0)

            # 4. Draw annotations
            output_frame = self._draw_annotations(frame, lights, vehicles)

            # 5. Violation state machine (crossing + red + dedupe)
            for v in vehicles:
                self._check_violation(v, frame, output_frame)

            # Free the frame buffer for vehicles that are done (already
            # violated, so no more OCR retries needed) to bound memory
            # on long/full-length videos.
            for vid in list(self.vehicle_frame_buffer.keys()):
                if vid in self.violated_vehicles:
                    del self.vehicle_frame_buffer[vid]

            # Track every vehicle ID ever seen (cheap: just integers) so
            # final statistics stay accurate even after the heavier
            # per-vehicle dicts below get pruned for old/stale IDs.
            for v in vehicles:
                self.all_vehicle_ids_seen.add(v['id'])
                self.vehicle_last_seen[v['id']] = self.frame_count

            # Long-video memory management: periodically drop bookkeeping
            # for vehicle IDs that have genuinely left the scene (not seen
            # for VEHICLE_STALE_AFTER_FRAMES+ frames). Without this, a
            # multi-hour video with thousands of distinct vehicles would
            # grow vehicle_history / vehicle_first_seen_side / the frame
            # buffer forever. A short grace period (rather than pruning
            # every frame) means a brief 1-2 frame occlusion never wipes
            # useful in-progress history.
            if self.frame_count % MEMORY_CLEANUP_EVERY_N_FRAMES == 0:
                stale_ids = [
                    vid for vid, last in self.vehicle_last_seen.items()
                    if self.frame_count - last > VEHICLE_STALE_AFTER_FRAMES
                ]
                for vid in stale_ids:
                    self.vehicle_frame_buffer.pop(vid, None)
                    self.vehicle_history.pop(vid, None)
                    self.vehicle_first_seen_side.pop(vid, None)
                    self.vehicle_last_seen.pop(vid, None)
                if stale_ids:
                    print(f"[memory cleanup] pruned {len(stale_ids)} stale vehicle track(s) "
                          f"at frame {self.frame_count}")

            # Shorter-interval cleanup specifically for the frame buffer
            # (the most memory-heavy structure, since it stores actual
            # image copies) so it doesn't wait as long as the full sweep.
            if self.frame_count % 30 == 0:
                for vid in list(self.vehicle_frame_buffer.keys()):
                    if self.frame_count - self.vehicle_last_seen.get(vid, 0) > 60:
                        del self.vehicle_frame_buffer[vid]

            writer.write(output_frame)

            if SHOW_WINDOW:
                cv2.imshow("Traffic Violation Detection", output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nUser stopped processing")
                    break
                elif key == ord('s'):
                    save_file = os.path.join(OUTPUT_FOLDER, f"frame_{self.frame_count}.jpg")
                    cv2.imwrite(save_file, frame)
                    print(f"Frame saved: {save_file}")
                elif key == ord(']'):
                    lane = self.lanes[self.active_lane_index]
                    lane["stop_line_y"] = min(self.frame_height - 1, lane["stop_line_y"] + 5)
                    print(f"{lane['name']} Stop Line Y -> {lane['stop_line_y']}")
                elif key == ord('['):
                    lane = self.lanes[self.active_lane_index]
                    lane["stop_line_y"] = max(0, lane["stop_line_y"] - 5)
                    print(f"{lane['name']} Stop Line Y -> {lane['stop_line_y']}")
                elif key == ord('n'):
                    self.active_lane_index = (self.active_lane_index + 1) % len(self.lanes)
                    print(f"Now adjusting: {self.lanes[self.active_lane_index]['name']}")

            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                lane_states = " ".join(f"{l['name']}={l['state']}" for l in self.lanes)
                print(f"Frame {self.frame_count} | Vehicles: {len(vehicles)} | "
                      f"Violations: {self.total_violations} | {lane_states} | "
                      f"Proc FPS: {self.frame_count/elapsed:.1f}")

            if TEST_MODE and self.frame_count >= MAX_TEST_FRAMES:
                print(f"\nTest mode: Processed {self.frame_count} frames")
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        report = self._save_report()

        elapsed = time.time() - self.start_time
        elapsed_minutes = elapsed / 60

        # Final summary - the headline numbers from the whole run
        print("\n" + "=" * 32)
        print("Video Finished")
        print(f"Vehicles Detected : {len(self.all_vehicle_ids_seen)}")
        print(f"Violations : {self.total_violations}")
        for lane in self.lanes:
            print(f"Red Frames ({lane['name']}) : {lane['red_frames']}")
        print(f"Processing Time : {elapsed_minutes:.1f} min")
        print(f"FPS : {self.frame_count/elapsed:.1f}" if elapsed > 0 else "FPS : n/a")
        print("=" * 32 + "\n")

        # Extra detail (kept alongside the required summary above)
        print(f"Frames Processed  : {self.frame_count}")
        print(f"Evidence Saved    : {self.total_violations}")
        print(f"CSV Saved         : {'Yes' if report else 'No'}")
        print(f"Output Video      : {output_video}")
        print(f"Evidence Folder   : {EVIDENCE_FOLDER}")
        if report:
            print(f"Report            : {report}")
        print()


# ======================================================================
# MAIN
# ======================================================================

if __name__ == "__main__":
    try:
        detector = TrafficViolationDetector()
        detector.process_video()
    except KeyboardInterrupt:
        print("\n\nUser interrupted with Ctrl+C")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    print("\nPress Enter to exit...")
    input()