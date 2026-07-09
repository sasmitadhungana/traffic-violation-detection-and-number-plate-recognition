"""
Red Light Violation Detection System
Complete working version with all features + fixes:
  1. Correct main-traffic-light selection (largest / most central box)
  2. Real crossing -> RED -> violation state machine per vehicle
  3. Frame-to-frame crossing comparison (not single-frame check)
  4. Duplicate-violation prevention (violated_vehicle_ids set)
  5. Plate detection scoped to each tracked vehicle's own box
  6. Full OCR pipeline (grayscale -> threshold -> resize -> EasyOCR)
  7. Evidence saving (frame / vehicle crop / plate crop)
  8. CSV violation log
  9. Plate text shown on video next to vehicle ID
  10. Improved on-screen UI (Signal / Vehicles / Violations / FPS / Time)
  11. Final run statistics printed at the end
  12. Auto-generated final report (CSV + annotated video + evidence folder)
  13. Cleaner OCR preprocessing
"""

import os
import sys
import csv
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

RED_CONFIRM_FRAMES = 5      # consecutive-ish red detections needed to "confirm" red
STATE_HISTORY_LEN = 10
STATE_VOTE_MIN = 3          # how many of the recent frames must agree

# Stop line position - ADJUST THESE FOR YOUR CAMERA.
# Item #4: the line no longer has to be a flat horizontal STOP_LINE_Y.
# Set STOP_LINE_P1 / STOP_LINE_P2 to two (x, y) points to match the road's
# actual perspective (an angled line across the lanes). If STOP_LINE_P2 is
# None, the system falls back to the old flat-horizontal behavior using
# STOP_LINE_Y, so existing configs keep working unchanged.
STOP_LINE_Y = 580
STOP_LINE_P1 = None   # e.g. (0, 560)     - left end of the stop line
STOP_LINE_P2 = None   # e.g. (1920, 610)  - right end of the stop line (angled)

# Item #4: instead of a single infinitely-thin line, treat crossing as
# happening once the vehicle passes this many pixels beyond the line -
# gives a small "stop line region/band" instead of a razor-thin trigger.
STOP_LINE_BAND_PX = 0

# How far (in px) a vehicle's box height/width must be to count (filters noise)
MIN_VEHICLE_SIZE = 30

# Item #3: ignore vehicles that barely move over a long window (parked
# cars, roadside vehicles the model mis-detects as "traffic"). A vehicle
# is treated as parked/irrelevant if its total displacement over the last
# PARKED_WINDOW_FRAMES frames is under PARKED_MOVEMENT_PX pixels.
PARKED_WINDOW_FRAMES = 40
PARKED_MOVEMENT_PX = 8

# Item #3: require the vehicle to actually be moving TOWARD the camera
# (downward in the frame, increasing Y) at the moment it crosses, not just
# any small jitter - filters out reversing vehicles and tracker noise.
MIN_FORWARD_SPEED_PX = 3

# Test mode - Set to False to process full video
TEST_MODE = True
MAX_TEST_FRAMES = 500

# Optional manual ROI override for the main traffic light (fix #8):
# set to (x1, y1, x2, y2) in pixel coordinates to force the system to only
# ever consider signals inside this box (e.g. the light directly over your
# lane). Leave as None to keep automatic largest-box selection.
TRAFFIC_LIGHT_ROI = None  # e.g. (1250, 260, 1360, 420)

# How many recent frames of each vehicle (image + box) to keep on hand so
# that, if a violation fires, OCR can be retried across several frames of
# that same vehicle instead of only the exact crossing frame (fix #1/#2/#3
# reliability improvement).
VEHICLE_FRAME_BUFFER_LEN = 8

# Item #5 (OCR issues): set True to run EasyOCR on GPU (much faster, and
# sometimes more accurate on higher-res crops). Requires a working CUDA
# install matching your PyTorch build. Falls back to CPU automatically if
# GPU init fails, so it's always safe to try True first.
OCR_USE_GPU = False

# Item #2 (stop line position): if True, the FIRST frame of the video is
# shown in a window where you click once on the stop line to set
# STOP_LINE_Y automatically, instead of guessing pixel values by hand.
CALIBRATE_STOP_LINE = False

# Item #3 (vehicle tracking stability): ultralytics' built-in tracker
# forgets a lost vehicle after `track_buffer` frames without a match. The
# default (30) can be too short when a vehicle is briefly hidden behind
# another one at a busy intersection, causing its ID to change. Raising
# this keeps the SAME ID through brief occlusions instead of creating a
# new track (which would let a violating vehicle dodge detection).
TRACK_BUFFER_FRAMES = 60
TRACKER_MATCH_THRESH = 0.75

# Item #12: mirror every console message into output/logs.txt as well,
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

try:
    import easyocr
    OCR_AVAILABLE = True
    print("EasyOCR available")
except ImportError:
    OCR_AVAILABLE = False
    print("EasyOCR not installed. Plate reading disabled.")
    print("   Install with: pip install easyocr")


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
        self.stop_line_y = STOP_LINE_Y

        # Vehicle tracking
        # vehicle_history[id] = list of (center_x, bottom_y) per frame seen
        self.vehicle_history = defaultdict(list)
        # vehicle_side[id] = "ABOVE" or "BELOW" relative to stop line, last known
        self.vehicle_side = {}
        self.violated_vehicles = set()
        self.total_violations = 0
        self.violation_records = []

        # fix #1/#2/#3 reliability: rolling buffer of each vehicle's own
        # recent (frame, box) pairs, so OCR can be retried across several
        # frames of the SAME vehicle rather than only the exact crossing
        # frame. Keeps the vehicle<->plate<->OCR link explicit and 1:1.
        self.vehicle_frame_buffer = defaultdict(list)
        self.vehicle_last_seen = {}

        # Traffic light state
        self.traffic_state = "UNKNOWN"
        self.state_history = []
        self.red_frames = 0
        self.red_confirmed = False
        self.main_light_box = None  # box of the light we trust this frame
        self.main_light_confidence = None

        # Initialize OCR
        self.ocr_reader = None
        if OCR_AVAILABLE:
            # Item #5 (OCR issues): try GPU first if requested, but never
            # let a bad/missing CUDA setup crash the whole program - fall
            # back to CPU automatically.
            if OCR_USE_GPU:
                try:
                    self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
                    print("OCR Reader initialized (GPU)")
                except Exception as e:
                    print(f"GPU OCR init failed ({e}), falling back to CPU")
                    self.ocr_reader = None

            if self.ocr_reader is None:
                try:
                    self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                    print("OCR Reader initialized (CPU)")
                except Exception as e:
                    print(f"OCR initialization failed: {e}")

        # Load models
        self._load_models()

        print(f"\nStop Line Y: {self.stop_line_y}")
        print(f"Vehicle Confidence: {VEHICLE_CONF}")
        print(f"Test Mode: {'ON' if TEST_MODE else 'OFF'}")
        if TEST_MODE:
            print(f"Max Frames: {MAX_TEST_FRAMES}")
        print("Controls: 'q'=quit, 's'=save frame, ']'/'['=nudge stop line down/up")
        print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    def _load_models(self):
        """Load YOLO models"""
        print("Loading models...")
        print("-" * 50)

        def _list_available(path):
            """Helper for item #1: when a model path is wrong, show what
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

        # Item #3: write a custom ByteTrack config with a longer track
        # buffer so a vehicle briefly hidden behind another one keeps its
        # SAME id instead of being re-assigned a new one when it re-appears.
        self.tracker_config_path = self._write_tracker_config()

    # ------------------------------------------------------------------
    def _write_tracker_config(self):
        """Item #3: generate a tuned bytetrack.yaml next to this script so
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
        Detect traffic lights and decide on the single MAIN signal that
        actually governs this stop line (fix #1).

        Strategy: gather every red/yellow/green candidate box above the
        candidate confidence floor, then pick the "main" one as the
        largest box (closest signal to the camera is usually the one
        controlling this lane) with ties broken by horizontal distance
        to the frame center.
        """
        if self.traffic_model is None:
            return []

        try:
            results = self.traffic_model.predict(
                frame,
                conf=TRAFFIC_CANDIDATE_CONF,
                verbose=False
            )
        except Exception:
            return []

        candidates = []
        frame_center_x = self.frame_width / 2 if self.frame_width else 0

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.traffic_model.names[class_id].lower()

                # normalize common label variants
                if "red" in label:
                    norm_label = "red"
                    color = (0, 0, 255)
                elif "green" in label:
                    norm_label = "green"
                    color = (0, 255, 0)
                elif "yellow" in label or "amber" in label:
                    norm_label = "yellow"
                    color = (0, 255, 255)
                else:
                    continue

                if confidence < TRAFFIC_ACCEPT_CONF:
                    continue

                area = max(0, x2 - x1) * max(0, y2 - y1)
                dist_to_center = abs(((x1 + x2) / 2) - frame_center_x)

                candidates.append({
                    'box': (x1, y1, x2, y2),
                    'color': color,
                    'label': norm_label,
                    'confidence': confidence,
                    'area': area,
                    'dist_to_center': dist_to_center
                })

        # fix #8: if a manual ROI is configured, restrict candidates to
        # signals whose center falls inside that ROI, so a fixed physical
        # light can be locked in regardless of what else the model sees.
        pool = candidates
        if TRAFFIC_LIGHT_ROI is not None:
            rx1, ry1, rx2, ry2 = TRAFFIC_LIGHT_ROI
            in_roi = [
                c for c in candidates
                if rx1 <= (c['box'][0] + c['box'][2]) / 2 <= rx2
                and ry1 <= (c['box'][1] + c['box'][3]) / 2 <= ry2
            ]
            if in_roi:
                pool = in_roi  # only trust lights inside the ROI when any exist there

        # Pick MAIN light: highest confidence first, ties broken by largest
        # area, then by closeness to frame center (fix #8).
        current_state = "UNKNOWN"
        self.main_light_box = None
        if pool:
            main = sorted(
                pool,
                key=lambda c: (-c['confidence'], -c['area'], c['dist_to_center'])
            )[0]
            current_state = main['label'].upper()
            self.main_light_box = main['box']
            self.main_light_confidence = main['confidence']

            if current_state == "RED":
                self.red_frames += 1

        # Update state history using ONLY the main light's state
        self.state_history.append(current_state)
        if len(self.state_history) > STATE_HISTORY_LEN:
            self.state_history.pop(0)

        # Confirm state by majority vote over recent history
        if len(self.state_history) >= 5:
            votes = defaultdict(int)
            for s in self.state_history:
                if s != "UNKNOWN":
                    votes[s] += 1

            if votes:
                most_common, count = max(votes.items(), key=lambda x: x[1])
                if count >= STATE_VOTE_MIN:
                    self.traffic_state = most_common
                    self.red_confirmed = (
                        most_common == "RED" and self.red_frames >= RED_CONFIRM_FRAMES
                    )
                    if most_common != "RED":
                        self.red_frames = 0

        return candidates

    # ------------------------------------------------------------------
    def _detect_vehicles(self, frame):
        """Detect and track vehicles"""
        if self.vehicle_model is None:
            return []

        try:
            # Item #3: use the tuned tracker config (longer track_buffer)
            # so IDs survive brief occlusions instead of switching.
            results = self.vehicle_model.track(
                frame,
                persist=True,
                conf=VEHICLE_CONF,
                tracker=self.tracker_config_path,
                verbose=False
            )
        except Exception:
            # Fall back to the default tracker if the custom config path
            # ever has a problem (e.g. ultralytics version mismatch).
            try:
                results = self.vehicle_model.track(
                    frame, persist=True, conf=VEHICLE_CONF, verbose=False
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
        (fix #5): we only ever search inside the padded crop of the
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
    def _read_plate_text(self, plate_image):
        """
        OCR pipeline (fix #6 / #13):
        grayscale -> denoise -> threshold -> resize -> EasyOCR

        Returns (text, confidence) so callers can compare reads taken
        from different frames of the same vehicle and keep the best one.
        A failed/empty read returns (None, 0.0).
        """
        if self.ocr_reader is None or plate_image is None or plate_image.size == 0:
            return None, 0.0

        try:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

            # Light denoise before threshold helps EasyOCR a lot on compressed CCTV video
            gray = cv2.bilateralFilter(gray, 7, 50, 50)

            # Adaptive threshold copes with uneven lighting on the plate
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 10
            )

            # Resize up for a cleaner OCR read
            h, w = thresh.shape
            if w > 0 and h > 0:
                scale = 300 / w
                new_w = 300
                new_h = max(1, int(h * scale))
                thresh = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            results = self.ocr_reader.readtext(
                thresh,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False
            )

            if results:
                # Prefer the longest reasonably-confident read; plates are
                # sometimes split into two boxes (e.g. "BA 2 CHA" / "4587")
                results = sorted(results, key=lambda r: r[0][0][0])  # left to right
                used = [r for r in results if r[2] >= 0.25]
                combined = "".join(r[1] for r in used).strip().replace(" ", "")
                avg_conf = sum(r[2] for r in used) / len(used) if used else 0.0

                if 2 <= len(combined) <= 12:
                    return combined, avg_conf

                best = max(results, key=lambda x: x[2])
                text = best[1].strip().replace(" ", "")
                if 2 <= len(text) <= 12:
                    return text, float(best[2])
        except Exception:
            pass

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
            # score combines plate-detector confidence and OCR confidence,
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
    def _stop_line_signed_distance(self, x, y):
        """
        Item #4: generalized stop-line check that supports either the old
        flat horizontal line (STOP_LINE_Y) or an angled two-point line
        (STOP_LINE_P1 / STOP_LINE_P2) that matches the road's real
        perspective. Returns a signed value that increases as (x, y) moves
        further past the line, in the "toward the camera" direction; 0 is
        exactly on the line.
        """
        if STOP_LINE_P1 is not None and STOP_LINE_P2 is not None:
            x1, y1 = STOP_LINE_P1
            x2, y2 = STOP_LINE_P2
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 or 1.0
            # signed perpendicular distance from the point to the line
            return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length
        return y - self.stop_line_y

    # ------------------------------------------------------------------
    def _update_crossing_state(self, vehicle):
        """
        Frame-to-frame crossing comparison (fix #3), generalized for an
        angled stop line and a configurable "band" thickness (fix #4):
        compares the vehicle's front-bumper point (bottom-center of its
        box) between the previous and current frame, rather than looking
        at only a single frame's position.

        Also requires the vehicle to be moving forward/toward the camera
        (fix #11) by at least MIN_FORWARD_SPEED_PX, which rules out
        tracker jitter or a vehicle rolling backward from being counted.

        Returns True exactly on the frame the vehicle transitions from
        BEFORE the stop line to AFTER it (past STOP_LINE_BAND_PX).
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

        prev_dist = self._stop_line_signed_distance(prev_x, prev_y)
        curr_dist = self._stop_line_signed_distance(curr_x, curr_y)

        crossed = prev_dist < STOP_LINE_BAND_PX <= curr_dist
        return crossed

    # ------------------------------------------------------------------
    def _is_parked_or_irrelevant(self, vehicle_id):
        """
        Item #3: treat a vehicle as parked/irrelevant (and skip violation
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
    def _check_violation(self, vehicle, frame, output_frame):
        """
        Core violation state machine (fix #2 / item #11 false-violation
        reduction). A violation requires ALL of:
            1. Signal is a CONFIRMED red
            2. Vehicle was before the stop line and crosses it (fix #4)
            3. Vehicle is moving forward, not jitter/reversing (fix #11)
            4. Vehicle has not already been recorded (dedupe, fix #4 orig)
            5. Vehicle isn't a parked/stationary false detection (item #3)
        """
        vehicle_id = vehicle['id']

        if vehicle_id in self.violated_vehicles:
            return False

        if self._is_parked_or_irrelevant(vehicle_id):
            return False

        crossed_now = self._update_crossing_state(vehicle)
        if not crossed_now:
            return False

        # The signal must be a CONFIRMED red at the moment of crossing
        if not self.red_confirmed:
            return False

        # ---- VIOLATION DETECTED ----
        self.violated_vehicles.add(vehicle_id)
        self.total_violations += 1

        # Vehicle -> Plate -> OCR, tried across every buffered frame of
        # THIS vehicle (fixes #1, #2, #3): far more reliable than reading
        # only the single exact crossing frame.
        best_plate_image, plate_text = self._best_plate_for_vehicle(
            vehicle_id, frame, vehicle['box']
        )

        plate_file = None
        if best_plate_image is not None and best_plate_image.size > 0:
            plate_file = os.path.join(EVIDENCE_FOLDER, f"plate_{vehicle_id}.jpg")
            cv2.imwrite(plate_file, best_plate_image)

        # Full annotated frame (fix #4/#9 naming: frame_<frame_number>.jpg)
        frame_file = os.path.join(EVIDENCE_FOLDER, f"frame_{self.frame_count}.jpg")
        cv2.imwrite(frame_file, output_frame)

        # Cropped vehicle (fix #4/#9 naming: vehicle_<id>.jpg)
        x1, y1, x2, y2 = vehicle['box']
        vehicle_crop = frame[y1:y2, x1:x2]
        vehicle_file = None
        if vehicle_crop.size > 0:
            vehicle_file = os.path.join(EVIDENCE_FOLDER, f"vehicle_{vehicle_id}.jpg")
            cv2.imwrite(vehicle_file, vehicle_crop)

        video_seconds = self.frame_count / self.fps if self.fps else 0
        video_timestamp = str(timedelta(seconds=int(video_seconds)))

        record = {
            'id': len(self.violation_records) + 1,
            'vehicle_id': vehicle_id,
            'frame': self.frame_count,
            'video_time': video_timestamp,
            'wall_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'plate_text': plate_text or "UNREADABLE",
            'state': self.traffic_state,
            'evidence_frame': frame_file,
            'evidence_vehicle': vehicle_file,
            'evidence_plate': plate_file
        }
        self.violation_records.append(record)

        print("\n" + "=" * 70)
        print(f"VIOLATION #{self.total_violations} DETECTED!")
        print("=" * 70)
        print(f"  Vehicle ID    : {vehicle_id}")
        print(f"  Frame         : {self.frame_count}")
        print(f"  Video Time    : {video_timestamp}")
        print(f"  Traffic State : {self.traffic_state}")
        print(f"  Plate Number  : {record['plate_text']}")
        print("-" * 70)
        print("  Evidence saved:")
        print(f"    Frame  : {os.path.basename(frame_file)}")
        if vehicle_file:
            print(f"    Vehicle: {os.path.basename(vehicle_file)}")
        if plate_file:
            print(f"    Plate  : {os.path.basename(plate_file)}")
        print("=" * 70 + "\n")

        return True

    # ------------------------------------------------------------------
    def _draw_annotations(self, frame, lights, vehicles):
        """Draw all annotations on frame"""
        output = frame.copy()

        # 1. Stop line (item #4: supports angled two-point line or flat line)
        line_color = (0, 0, 255) if self.red_confirmed else (0, 255, 255)

        if STOP_LINE_P1 is not None and STOP_LINE_P2 is not None:
            p1, p2 = STOP_LINE_P1, STOP_LINE_P2
            cv2.line(output, p1, p2, line_color, 4)
            label_x, label_y = p1[0], p1[1]
        else:
            cv2.line(output, (0, self.stop_line_y), (self.frame_width, self.stop_line_y), line_color, 4)
            for x in range(20, self.frame_width, 50):
                cv2.line(output, (x, self.stop_line_y - 4), (x + 25, self.stop_line_y - 4), line_color, 2)
            label_x, label_y = 20, self.stop_line_y

        text = "STOP LINE"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(output, (label_x - 10, label_y - text_size[1] - 15),
                      (label_x + text_size[0] + 10, label_y + 10), (0, 0, 0), -1)
        cv2.putText(output, text, (label_x, label_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        # 2. Traffic light status (based on confirmed / voted main-light state)
        if self.red_confirmed or self.traffic_state == "RED":
            status = "RED LIGHT"
            status_color = (0, 0, 255)
            bg_color = (0, 0, 100)
        elif self.traffic_state == "GREEN":
            status = "GREEN LIGHT"
            status_color = (0, 255, 0)
            bg_color = (0, 80, 0)
        elif self.traffic_state == "YELLOW":
            status = "YELLOW LIGHT"
            status_color = (0, 255, 255)
            bg_color = (80, 80, 0)
        else:
            status = "NO SIGNAL"
            status_color = (255, 255, 0)
            bg_color = (80, 80, 0)

        cv2.rectangle(output, (10, 10), (460, 90), (0, 0, 0), -1)
        cv2.rectangle(output, (12, 12), (458, 88), bg_color, 2)
        cv2.putText(output, f"TRAFFIC: {status}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        conf_txt = f"{self.main_light_confidence:.2f}" if self.main_light_confidence else "-"
        cv2.putText(output, f"Red Frames: {self.red_frames}   Confidence: {conf_txt}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        video_seconds = self.frame_count / self.fps if self.fps else 0
        video_timestamp = str(timedelta(seconds=int(video_seconds)))

        # 3. Draw traffic lights (highlight the MAIN one used for decisions)
        for light in lights:
            x1, y1, x2, y2 = light['box']
            is_main = (self.main_light_box == light['box'])
            thickness = 3 if is_main else 1
            cv2.rectangle(output, (x1, y1), (x2, y2), light['color'], thickness)
            label_txt = light['label'].upper() + (" [MAIN]" if is_main else "")
            cv2.putText(output, label_txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, light['color'], 2)

        # 4. Draw vehicles
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

            id_text = f"ID:{vehicle_id}  {plate_text}" if plate_text else f"ID:{vehicle_id}"

            cv2.rectangle(output, (x1, y1 - 28), (x1 + 220, y1), (0, 0, 0), -1)
            cv2.putText(output, id_text, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 5. Statistics bar (item #9: added total tracked vehicles + last violation info)
        elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        proc_fps = self.frame_count / elapsed if elapsed > 0 else 0

        cv2.rectangle(output, (10, 95), (560, 250), (0, 0, 0), -1)

        y = 118
        cv2.putText(output, f"Frame: {self.frame_count}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Vehicles: {len(vehicles)}", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Violations: {self.total_violations}", (360, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if self.total_violations > 0 else (255, 255, 255), 1)

        y = 145
        cv2.putText(output, f"FPS: {proc_fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Signal: {self.traffic_state}", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        cv2.putText(output, f"Stop Y: {self.stop_line_y}", (360, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y = 172
        cv2.putText(output, f"Time: {video_timestamp}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Total Tracked: {len(self.vehicle_history)}", (200, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y = 199
        if self.violation_records:
            last = self.violation_records[-1]
            last_txt = f"Last Violation: ID {last['vehicle_id']}  {last['plate_text']}"
        else:
            last_txt = "Last Violation: none yet"
        cv2.putText(output, last_txt, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

        cv2.putText(output, "q=quit  s=save  [/]=move stop line", (10, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output

    # ------------------------------------------------------------------
    def _save_report(self):
        """Generate CSV report (fix #8 / #12)"""
        # Item #12 (final output structure): the primary report lives
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
                # Column order matches the requested report format:
                # Vehicle ID | Plate | Frame | Time | Signal (+ extra detail columns after)
                writer.writerow([
                    "Vehicle ID", "Plate", "Frame", "Time", "Signal",
                    "Violation ID", "Wall Time",
                    "Evidence Frame", "Evidence Vehicle", "Evidence Plate"
                ])
                for r in self.violation_records:
                    writer.writerow([
                        r['vehicle_id'],
                        r['plate_text'],
                        r['frame'],
                        r['video_time'],
                        r['state'],
                        r['id'],
                        r['wall_time'],
                        os.path.basename(r['evidence_frame']) if r['evidence_frame'] else "",
                        os.path.basename(r['evidence_vehicle']) if r['evidence_vehicle'] else "",
                        os.path.basename(r['evidence_plate']) if r['evidence_plate'] else ""
                    ])

        print(f"\nReport saved: {report_file}")
        print(f"Timestamped copy: {timestamped_file}")
        return report_file

    # ------------------------------------------------------------------
    def _calibrate_stop_line(self, frame):
        """
        Item #2 (stop line position): shows the first frame in a window
        and lets you click once on the stop line. Returns the clicked Y
        so you don't have to guess STOP_LINE_Y by hand. Press any key to
        confirm after clicking, or just press a key without clicking to
        keep the current STOP_LINE_Y.
        """
        picked = {'y': self.stop_line_y}
        window = "Click the stop line, then press any key"

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                picked['y'] = y

        preview = frame.copy()
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, on_click)

        while True:
            display = preview.copy()
            cv2.line(display, (0, picked['y']), (self.frame_width, picked['y']), (0, 255, 255), 3)
            cv2.putText(display, f"Stop Line Y = {picked['y']}  (click to move, any key to confirm)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow(window, display)
            if cv2.waitKey(30) != -1:
                break

        cv2.destroyWindow(window)
        print(f"Stop line calibrated to Y = {picked['y']}")
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

        # Item #2 (stop line position): optional one-click calibration on
        # the first frame instead of guessing STOP_LINE_Y by hand.
        if CALIBRATE_STOP_LINE:
            ret, calib_frame = cap.read()
            if ret:
                self.stop_line_y = self._calibrate_stop_line(calib_frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start after calibrating

        print("\n" + "=" * 60)
        print("STARTING VIDEO PROCESSING")
        print("=" * 60)
        print(f"Video: {os.path.basename(VIDEO_FILE)}")
        print(f"Size: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps:.1f}")
        print(f"Stop Line: Y={self.stop_line_y}")
        print("=" * 60 + "\n")

        output_video = os.path.join(OUTPUT_FOLDER, "tracking_output.mp4")
        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height)
        )

        self.start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached")
                break

            self.frame_count += 1

            # 1. Traffic light state (main-light selection fix)
            lights = self._detect_traffic_light(frame)

            # 2. Vehicles
            vehicles = self._detect_vehicles(frame)

            # 3. Update per-vehicle position history
            for v in vehicles:
                vid = v['id']
                self.vehicle_history[vid].append((v['center_x'], v['bottom_y']))
                if len(self.vehicle_history[vid]) > 20:
                    self.vehicle_history[vid].pop(0)

                # Keep a short rolling buffer of this vehicle's own raw
                # frame + box so a later violation can retry plate OCR
                # across several frames of the SAME vehicle (fix #1/#2/#3).
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

            # Also drop buffers for vehicles that have genuinely left the
            # scene (not seen for 60+ frames, so a brief one-frame
            # occlusion won't wipe useful history) so long videos with
            # many passing vehicles don't leak memory.
            for v in vehicles:
                self.vehicle_last_seen[v['id']] = self.frame_count
            if self.frame_count % 30 == 0:
                for vid in list(self.vehicle_frame_buffer.keys()):
                    if self.frame_count - self.vehicle_last_seen.get(vid, 0) > 60:
                        del self.vehicle_frame_buffer[vid]

            writer.write(output_frame)
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
                self.stop_line_y = min(self.frame_height - 1, self.stop_line_y + 5)
                print(f"Stop Line Y -> {self.stop_line_y}")
            elif key == ord('['):
                self.stop_line_y = max(0, self.stop_line_y - 5)
                print(f"Stop Line Y -> {self.stop_line_y}")

            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                print(f"Frame {self.frame_count} | Vehicles: {len(vehicles)} | "
                      f"Violations: {self.total_violations} | State: {self.traffic_state} | "
                      f"Proc FPS: {self.frame_count/elapsed:.1f}")

            if TEST_MODE and self.frame_count >= MAX_TEST_FRAMES:
                print(f"\nTest mode: Processed {self.frame_count} frames")
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        report = self._save_report()

        elapsed = time.time() - self.start_time

        # fix #10: final summary in the exact requested format
        print("\n" + "=" * 32)
        print("Processing Complete")
        print(f"Frames : {self.frame_count}")
        print(f"Vehicles : {len(self.vehicle_history)}")
        print(f"Violations : {self.total_violations}")
        print(f"Evidence Saved : {self.total_violations}")
        print(f"CSV Saved : {'Yes' if report else 'No'}")
        print("=" * 32 + "\n")

        # Extra detail (kept alongside the required summary above)
        print(f"Processing Time   : {elapsed:.1f} seconds")
        if elapsed > 0:
            print(f"Average FPS       : {self.frame_count/elapsed:.1f}")
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