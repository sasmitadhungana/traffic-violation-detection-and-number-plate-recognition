

import os
import sys
import csv
import time
import pickle
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION - Edit these to match your setup
# ============================================================================

# Paths to your models
MODEL_PATHS = {
    'traffic': os.path.join(os.path.dirname(__file__), "models", "traffic_best.pt"),
    'vehicle': os.path.join(os.path.dirname(__file__), "models", "vehicle_best.pt"),
    'plate': os.path.join(os.path.dirname(__file__), "models", "plate_best.pt"),
}

# Path to your video file
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "videos", "testvideo", "test7min.mp4")

# Output directories
OUTPUT_DIRS = {
    'base': os.path.join(os.path.dirname(__file__), "output"),
    'evidence': os.path.join(os.path.dirname(__file__), "output", "violations"),
    'plates': os.path.join(os.path.dirname(__file__), "output", "violations", "plates"),
    'reports': os.path.join(os.path.dirname(__file__), "output", "reports"),
    'checkpoints': os.path.join(os.path.dirname(__file__), "output", "checkpoints"),
}

# Create output directories
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# DETECTION PARAMETERS
# ============================================================================

class DetectionConfig:
    """Configuration for detection models"""
    
    # Confidence thresholds
    TRAFFIC_CANDIDATE_CONF = 0.10  # Low threshold to catch all candidates
    TRAFFIC_ACCEPT_CONF = 0.20     # Higher threshold for reliable detections
    VEHICLE_CONF = 0.30
    PLATE_CONF = 0.25              # Slightly lower to catch more plates
    
    # Model settings
    DETECTION_IMGSZ = 640          # Image size for YOLO inference
    INFERENCE_DEVICE = "cpu"       # "cuda" or "cpu"
    HALF_PRECISION = False         # FP16 inference (GPU only)
    
    # Traffic light tracking
    TRAFFIC_SKIP_FRAMES = 2        # Don't detect lights every frame
    RED_CONFIRM_FRAMES = 5         # Frames needed to confirm red
    STATE_HISTORY_LEN = 10
    STATE_VOTE_MIN = 3

class TrackerConfig:
    """ByteTrack configuration for vehicle tracking"""
    
    TRACK_BUFFER = 90              # How many frames to keep lost tracks
    MATCH_THRESH = 0.65            # IoU threshold for matching
    HIGH_THRESH = 0.45             # High confidence threshold
    LOW_THRESH = 0.1               # Low confidence threshold
    NEW_TRACK_THRESH = 0.55        # Threshold for new tracks

class OCRConfig:
    """OCR configuration"""
    
    USE_GPU = False
    MIN_PLATE_LENGTH = 2
    MAX_PLATE_LENGTH = 12
    MIN_OCR_CONFIDENCE = 0.3

class VideoConfig:
    """Video processing configuration"""
    
    SHOW_PREVIEW = True
    TEST_MODE = False
    MAX_TEST_FRAMES = 500
    
    # Long video handling
    MEMORY_CLEANUP_INTERVAL = 200
    STALE_VEHICLE_FRAMES = 300
    CHECKPOINT_INTERVAL = 1000
    ENABLE_CHECKPOINTS = True
    FRAME_READ_RETRIES = 5
    
    # Evidence saving
    SAVE_INTERMEDIATE = True
    VEHICLE_FRAME_BUFFER = 8

# ============================================================================
# LANE CONFIGURATION
# ============================================================================


LANES = [
    {
        "name": "Lane 1 - Left Turn",
        "x_range": (0, 640),
        "stop_line_y": 780,           # Bottom of frame - in front of vehicles
        "light_roi": (450, 120, 550, 220),  # Top of frame - across intersection
        "p1": (50, 775),
        "p2": (590, 785),
    },
    {
        "name": "Lane 2 - Straight",
        "x_range": (640, 1280),
        "stop_line_y": 785,
        "light_roi": (900, 110, 1000, 210),
        "p1": (640, 780),
        "p2": (1280, 790),
    },
    {
        "name": "Lane 3 - Right Turn",
        "x_range": (1280, 1750),
        "stop_line_y": 790,
        "light_roi": (1350, 110, 1450, 210),
        "p1": (1280, 785),
        "p2": (1700, 795),
    },
    {
        "name": "Lane 4 - Near Side",
        "x_range": (1750, 1920),
        "stop_line_y": 820,           # Closest lane - highest Y value
        "light_roi": None,            # Shares light with Lane 3
        "p1": (1750, 815),
        "p2": (1900, 825),
    },
]

# ============================================================================
# OCR IMPLEMENTATION
# ============================================================================

class LicensePlateOCR:
    """
    Enhanced OCR for license plate reading.
    Uses PaddleOCR with multiple preprocessing techniques.
    """
    
    def __init__(self, use_gpu: bool = False):
        self.reader = None
        self._init_ocr(use_gpu)
    
    def _init_ocr(self, use_gpu: bool):
        """Initialize PaddleOCR with error handling"""
        try:
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5,
                rec_algorithm='SVTR_LCNet',
                max_text_length=12,
            )
            print(f"✅ OCR initialized (GPU={use_gpu})")
        except ImportError:
            print("⚠️ PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        except Exception as e:
            print(f"❌ OCR initialization failed: {e}")
    
    def preprocess_plate(self, image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Apply multiple preprocessing techniques to improve OCR accuracy.
        Returns list of (name, processed_image) pairs.
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        processed = []
        
        # 1. Original grayscale
        processed.append(("original", gray))
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed.append(("clahe", clahe.apply(gray)))
        
        # 3. Denoised + CLAHE
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        processed.append(("denoised", clahe.apply(denoised)))
        
        # 4. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )
        processed.append(("adaptive", adaptive))
        
        # 5. Otsu threshold
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed.append(("otsu", otsu))
        
        # 6. Sharpened
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed.append(("sharpened", cv2.filter2D(gray, -1, kernel)))
        
        # 7. Upscaled (for small plates)
        h, w = gray.shape
        if h < 60 or w < 120:
            scale = max(2.0, 300.0 / w)
            upscaled = cv2.resize(gray, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)
            processed.append(("upscaled", upscaled))
        
        return processed
    
    def read_plate(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Read license plate text with multiple preprocessing attempts.
        Returns (text, confidence) or (None, 0.0) if no text found.
        """
        if self.reader is None or image is None or image.size == 0:
            return None, 0.0
        
        best_text = None
        best_confidence = 0.0
        
        for name, processed_img in self.preprocess_plate(image):
            try:
                # Convert to RGB for PaddleOCR
                if len(processed_img.shape) == 2:
                    rgb_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                else:
                    rgb_img = processed_img
                
                # Run OCR
                results = self.reader.ocr(rgb_img, cls=True)
                
                if not results or not results[0]:
                    continue
                
                # Extract and clean results
                text_parts = []
                confidences = []
                
                for line in results[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        
                        # Clean text - keep alphanumeric characters
                        cleaned = ''.join(c for c in text if c.isalnum())
                        cleaned = cleaned.strip().upper()
                        
                        if cleaned and len(cleaned) >= OCRConfig.MIN_PLATE_LENGTH:
                            text_parts.append(cleaned)
                            confidences.append(confidence)
                
                if text_parts:
                    combined_text = ''.join(text_parts)
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # Validate the result
                    if len(combined_text) <= OCRConfig.MAX_PLATE_LENGTH:
                        has_letter = any(c.isalpha() for c in combined_text)
                        has_number = any(c.isdigit() for c in combined_text)
                        
                        # Short plates should have both letters and numbers
                        if len(combined_text) >= 5 or (has_letter and has_number):
                            if avg_confidence > best_confidence:
                                best_text = combined_text
                                best_confidence = avg_confidence
                                
            except Exception:
                continue
        
        return best_text, best_confidence

# ============================================================================
# MAIN DETECTOR CLASS
# ============================================================================

class TrafficViolationDetector:
    """
    Main class for traffic violation detection.
    Handles vehicle tracking, traffic light detection, violation detection,
    and evidence collection.
    """
    
    def __init__(self):
        """Initialize the detector with all components"""
        print("\n" + "=" * 70)
        print("🚦 TRAFFIC VIOLATION DETECTION SYSTEM")
        print("=" * 70)
        
        # State tracking
        self.frame_count = 0
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.total_frames = None
        self.start_time = None
        
        # Vehicle tracking dictionaries
        self.vehicle_history = defaultdict(list)          # Position history per vehicle
        self.vehicle_first_seen_side = {}                # Which side of line first seen
        self.vehicle_last_seen = {}                      # Last frame seen
        self.vehicle_frame_buffer = defaultdict(list)    # Recent frames per vehicle
        self.violated_vehicles = set()                   # Already ticketed vehicles
        
        # Statistics
        self.all_vehicle_ids = set()
        self.total_violations = 0
        self.violation_records = []
        self.saved_plates = {}
        
        # Lane configuration
        self.lanes = self._setup_lanes()
        self.active_lane_index = 0
        
        # Traffic light cache (skip frames optimization)
        self._cached_traffic_lights = []
        
        # Initialize components
        self.ocr = LicensePlateOCR(use_gpu=OCRConfig.USE_GPU)
        self._load_models()
        self._setup_tracker()
        
        # Print configuration summary
        self._print_config_summary()
    
    def _setup_lanes(self) -> List[Dict]:
        """Initialize lane configurations with state tracking"""
        lanes = []
        for cfg in LANES:
            lanes.append({
                **cfg,
                "state": "UNKNOWN",
                "state_history": [],
                "red_frames": 0,
                "red_confirmed": False,
                "main_box": None,
                "confidence": None,
            })
        return lanes
    
    def _load_models(self):
        """Load YOLO models with error handling"""
        print("\n📦 Loading Models...")
        print("-" * 50)
        
        self.models = {}
        for name, path in MODEL_PATHS.items():
            if os.path.exists(path):
                try:
                    self.models[name] = YOLO(path)
                    print(f"  ✅ {name.title()}: {os.path.basename(path)}")
                except Exception as e:
                    print(f"  ❌ {name.title()}: Failed - {e}")
                    self.models[name] = None
            else:
                print(f"  ⚠️ {name.title()}: Not found at {path}")
                self.models[name] = None
        
        print("-" * 50)
        
        # Check if we have the minimum required models
        if self.models.get('vehicle') is None:
            print("\n❌ ERROR: Vehicle model is required!")
            sys.exit(1)
    
    def _setup_tracker(self):
        """Create custom ByteTrack configuration"""
        config_path = os.path.join(os.path.dirname(__file__), "custom_bytetrack.yaml")
        
        config_content = f"""
tracker_type: bytetrack
track_high_thresh: {TrackerConfig.HIGH_THRESH}
track_low_thresh: {TrackerConfig.LOW_THRESH}
new_track_thresh: {TrackerConfig.NEW_TRACK_THRESH}
track_buffer: {TrackerConfig.TRACK_BUFFER}
match_thresh: {TrackerConfig.MATCH_THRESH}
fuse_score: True
"""
        try:
            with open(config_path, "w") as f:
                f.write(config_content)
            self.tracker_config = config_path
            print(f"✅ Tracker configured (buffer={TrackerConfig.TRACK_BUFFER})")
        except Exception as e:
            print(f"⚠️ Using default tracker: {e}")
            self.tracker_config = "bytetrack.yaml"
    
    def _print_config_summary(self):
        """Print a summary of the configuration"""
        print("\n📋 Configuration Summary")
        print("-" * 50)
        print(f"  Lanes: {len(self.lanes)}")
        for lane in self.lanes:
            line_desc = f"Y={lane['stop_line_y']}"
            if lane.get('p1') and lane.get('p2'):
                line_desc = f"({lane['p1']} → {lane['p2']})"
            print(f"    • {lane['name']}: {line_desc}")
        print(f"  Vehicle Confidence: {DetectionConfig.VEHICLE_CONF}")
        print(f"  Plate Confidence: {DetectionConfig.PLATE_CONF}")
        print(f"  Track Buffer: {TrackerConfig.TRACK_BUFFER} frames")
        print(f"  Checkpoint Interval: {VideoConfig.CHECKPOINT_INTERVAL} frames")
        print("=" * 70 + "\n")
    
    # ========================================================================
    # LANE AND DISTANCE UTILITIES
    # ========================================================================
    
    def _get_lane_for_position(self, x: int) -> Dict:
        """Determine which lane a vehicle belongs to based on x position"""
        for lane in self.lanes:
            x_min, x_max = lane["x_range"]
            if x_min <= x <= x_max:
                return lane
        
        # Fallback: closest lane by center
        return min(self.lanes, key=lambda l: abs((l["x_range"][0] + l["x_range"][1]) / 2 - x))
    
    def _distance_to_stop_line(self, lane: Dict, x: int, y: int) -> float:
        """
        Calculate signed distance from point to stop line.
        Positive = past the line (toward intersection).
        """
        if lane.get("p1") is not None and lane.get("p2") is not None:
            x1, y1 = lane["p1"]
            x2, y2 = lane["p2"]
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if length == 0:
                return y - lane["stop_line_y"]
            return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length
        return y - lane["stop_line_y"]
    
    # ========================================================================
    # DETECTION METHODS
    # ========================================================================
    
    def detect_traffic_lights(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect traffic lights and update lane states.
        Uses caching to avoid running detection every frame.
        """
        if self.models.get('traffic') is None:
            return []
        
        # Skip frames optimization
        if (DetectionConfig.TRAFFIC_SKIP_FRAMES > 1 and 
            self.frame_count % DetectionConfig.TRAFFIC_SKIP_FRAMES != 0):
            candidates = self._cached_traffic_lights
        else:
            # Run detection
            try:
                results = self.models['traffic'].predict(
                    frame,
                    conf=DetectionConfig.TRAFFIC_CANDIDATE_CONF,
                    imgsz=DetectionConfig.DETECTION_IMGSZ,
                    device=DetectionConfig.INFERENCE_DEVICE,
                    half=DetectionConfig.HALF_PRECISION,
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
                    label = self.models['traffic'].names[class_id].lower()
                    
                    # Map class name to color
                    if "red" in label:
                        label_type, color = "red", (0, 0, 255)
                    elif "green" in label:
                        label_type, color = "green", (0, 255, 0)
                    elif "yellow" in label or "amber" in label:
                        label_type, color = "yellow", (0, 255, 255)
                    else:
                        continue
                    
                    if confidence < DetectionConfig.TRAFFIC_ACCEPT_CONF:
                        continue
                    
                    candidates.append({
                        'box': (x1, y1, x2, y2),
                        'color': color,
                        'label': label_type,
                        'confidence': confidence,
                        'center_x': (x1 + x2) / 2,
                        'center_y': (y1 + y2) / 2,
                    })
            
            self._cached_traffic_lights = candidates
        
        # Update each lane's state
        for lane in self.lanes:
            roi = lane.get("light_roi")
            if roi:
                rx1, ry1, rx2, ry2 = roi
                pool = [
                    c for c in candidates
                    if rx1 <= c['center_x'] <= rx2 and ry1 <= c['center_y'] <= ry2
                ]
            else:
                pool = candidates
            
            # Determine current state
            current_state = "UNKNOWN"
            lane["main_box"] = None
            if pool:
                best = max(pool, key=lambda c: c['confidence'])
                current_state = best['label'].upper()
                lane["main_box"] = best['box']
                lane["confidence"] = best['confidence']
                if current_state == "RED":
                    lane["red_frames"] += 1
            
            # Update history
            lane["state_history"].append(current_state)
            if len(lane["state_history"]) > DetectionConfig.STATE_HISTORY_LEN:
                lane["state_history"].pop(0)
            
            # Vote on state from history
            if len(lane["state_history"]) >= 5:
                votes = defaultdict(int)
                for s in lane["state_history"]:
                    if s != "UNKNOWN":
                        votes[s] += 1
                if votes:
                    most_common, count = max(votes.items(), key=lambda x: x[1])
                    if count >= DetectionConfig.STATE_VOTE_MIN:
                        lane["state"] = most_common
                        lane["red_confirmed"] = (
                            most_common == "RED" and 
                            lane["red_frames"] >= DetectionConfig.RED_CONFIRM_FRAMES
                        )
                        if most_common != "RED":
                            lane["red_frames"] = 0
        
        return candidates
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect and track vehicles using YOLO + ByteTrack.
        Returns list of vehicle dictionaries with tracking IDs.
        """
        if self.models.get('vehicle') is None:
            return []
        
        try:
            results = self.models['vehicle'].track(
                frame,
                persist=True,
                conf=DetectionConfig.VEHICLE_CONF,
                imgsz=DetectionConfig.DETECTION_IMGSZ,
                device=DetectionConfig.INFERENCE_DEVICE,
                half=DetectionConfig.HALF_PRECISION,
                tracker=self.tracker_config,
                verbose=False
            )
        except Exception:
            # Fallback to default tracker
            try:
                results = self.models['vehicle'].track(
                    frame,
                    persist=True,
                    conf=DetectionConfig.VEHICLE_CONF,
                    imgsz=DetectionConfig.DETECTION_IMGSZ,
                    verbose=False
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
                
                # Filter out small detections
                if (y2 - y1) < 30 or (x2 - x1) < 30:
                    continue
                
                vehicles.append({
                    'id': vehicle_id,
                    'box': (x1, y1, x2, y2),
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2,
                    'bottom_y': y2,
                    'confidence': confidence,
                })
        
        return vehicles
    
    def detect_license_plate(self, frame: np.ndarray, vehicle_box: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Detect license plate within a vehicle's bounding box.
        Uses multi-scale detection for better results.
        """
        if self.models.get('plate') is None:
            return None
        
        x1, y1, x2, y2 = vehicle_box
        padding = 20
        
        # Crop vehicle region with padding
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(self.frame_width, x2 + padding)
        y2p = min(self.frame_height, y2 + padding)
        
        vehicle_region = frame[y1p:y2p, x1p:x2p]
        if vehicle_region.size == 0:
            return None
        
        # Try multiple scales for better detection
        scales = [1.0, 1.5, 0.8]
        best_plate = None
        best_confidence = 0.0
        
        for scale in scales:
            if scale != 1.0:
                h, w = vehicle_region.shape[:2]
                new_w = int(w * scale)
                new_h = int(h * scale)
                if new_w < 50 or new_h < 50:
                    continue
                scaled_region = cv2.resize(vehicle_region, (new_w, new_h), cv2.INTER_CUBIC)
            else:
                scaled_region = vehicle_region
            
            try:
                results = self.models['plate'].predict(
                    scaled_region,
                    conf=DetectionConfig.PLATE_CONF,
                    device=DetectionConfig.INFERENCE_DEVICE,
                    half=DetectionConfig.HALF_PRECISION,
                    verbose=False
                )
            except Exception:
                continue
            
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    if confidence > best_confidence:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        
                        # Scale back if needed
                        if scale != 1.0:
                            px1 = int(px1 / scale)
                            py1 = int(py1 / scale)
                            px2 = int(px2 / scale)
                            py2 = int(py2 / scale)
                        
                        # Convert to absolute coordinates
                        abs_x1 = x1p + px1
                        abs_y1 = y1p + py1
                        abs_x2 = x1p + px2
                        abs_y2 = y1p + py2
                        
                        plate_image = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                        if plate_image.size > 0:
                            best_plate = {
                                'image': plate_image,
                                'confidence': confidence,
                                'box': (abs_x1, abs_y1, abs_x2, abs_y2),
                            }
                            best_confidence = confidence
        
        return best_plate
    
    # ========================================================================
    # VEHICLE TRACKING AND VIOLATION DETECTION
    # ========================================================================
    
    def update_vehicle_history(self, vehicles: List[Dict], frame: np.ndarray):
        """Update vehicle position history and frame buffer"""
        for vehicle in vehicles:
            vid = vehicle['id']
            
            # Check which side of stop line first seen
            if vid not in self.vehicle_first_seen_side:
                lane = self._get_lane_for_position(vehicle['center_x'])
                dist = self._distance_to_stop_line(
                    lane, vehicle['center_x'], vehicle['bottom_y']
                )
                self.vehicle_first_seen_side[vid] = "BEFORE" if dist < 0 else "AFTER"
            
            # Update position history
            self.vehicle_history[vid].append((vehicle['center_x'], vehicle['bottom_y']))
            if len(self.vehicle_history[vid]) > 20:
                self.vehicle_history[vid].pop(0)
            
            # Update frame buffer for OCR
            buffer = self.vehicle_frame_buffer[vid]
            buffer.append({'frame': frame.copy(), 'box': vehicle['box']})
            if len(buffer) > VideoConfig.VEHICLE_FRAME_BUFFER:
                buffer.pop(0)
            
            # Track last seen
            self.vehicle_last_seen[vid] = self.frame_count
            self.all_vehicle_ids.add(vid)
    
    def check_red_light_violation(self, vehicle: Dict, frame: np.ndarray, 
                                   output_frame: np.ndarray) -> bool:
        """
        Check if a vehicle committed a red light violation.
        Returns True if violation was detected and processed.
        """
        vid = vehicle['id']
        
        # Already ticketed
        if vid in self.violated_vehicles:
            return False
        
        # Must have been seen approaching from before the line
        if self.vehicle_first_seen_side.get(vid) != "BEFORE":
            return False
        
        # Check if vehicle is parked/stationary
        if self._is_parked(vid):
            return False
        
        # Check if vehicle crossed the stop line this frame
        if not self._crossed_stop_line(vehicle):
            return False
        
        # Check if lane light is red
        lane = self._get_lane_for_position(vehicle['center_x'])
        if not lane["red_confirmed"]:
            return False
        
        # ─── VIOLATION DETECTED ───
        return self._process_violation(vehicle, frame, output_frame, lane)
    
    def _is_parked(self, vehicle_id: int) -> bool:
        """Check if vehicle has been stationary for too long"""
        history = self.vehicle_history.get(vehicle_id, [])
        if len(history) < 40:
            return False
        
        window = history[-40:]
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        displacement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return displacement < 8
    
    def _crossed_stop_line(self, vehicle: Dict) -> bool:
        """Check if vehicle crossed the stop line this frame"""
        vid = vehicle['id']
        history = self.vehicle_history.get(vid, [])
        
        if len(history) < 2:
            return False
        
        prev_x, prev_y = history[-2]
        curr_x, curr_y = history[-1]
        
        # Must be moving forward (toward camera = increasing y)
        if (curr_y - prev_y) < 3:
            return False
        
        lane = self._get_lane_for_position(curr_x)
        prev_dist = self._distance_to_stop_line(lane, prev_x, prev_y)
        curr_dist = self._distance_to_stop_line(lane, curr_x, curr_y)
        
        # Crossed from before line (negative) to after line (positive)
        return prev_dist < 0 <= curr_dist
    
    def _process_violation(self, vehicle: Dict, frame: np.ndarray,
                          output_frame: np.ndarray, lane: Dict) -> bool:
        """Process a detected violation - save evidence and update records"""
        vid = vehicle['id']
        self.violated_vehicles.add(vid)
        self.total_violations += 1
        
        # Get best license plate from frame buffer
        plate_image, plate_text, ocr_confidence = self._get_best_plate(vid, frame, vehicle['box'])
        
        # Save plate if detected
        if plate_image is not None and plate_image.size > 0:
            plate_path = os.path.join(
                OUTPUT_DIRS['plates'],
                f"plate_violation_{self.total_violations:03d}_id_{vid}.jpg"
            )
            cv2.imwrite(plate_path, plate_image)
            self.saved_plates[vid] = plate_path
        else:
            plate_path = None
        
        # Save evidence
        evidence = self._save_evidence(
            output_frame, frame, vehicle, plate_image, plate_text
        )
        
        # Create record
        video_time = str(timedelta(seconds=int(self.frame_count / self.fps)))
        record = {
            'violation_id': self.total_violations,
            'vehicle_id': vid,
            'lane': lane['name'],
            'frame': self.frame_count,
            'video_time': video_time,
            'plate_text': plate_text or "UNREADABLE",
            'ocr_confidence': ocr_confidence,
            'signal_state': lane['state'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **evidence,
            'plate_path': plate_path,
        }
        
        self.violation_records.append(record)
        self._save_violation_to_csv(record)
        
        # Print violation details
        self._print_violation(record)
        
        return True
    
    def _get_best_plate(self, vehicle_id: int, fallback_frame: np.ndarray,
                        fallback_box: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """
        Get the best license plate detection from vehicle's frame buffer.
        Tries multiple frames for better OCR results.
        """
        # Get frames from buffer or use fallback
        buffer = self.vehicle_frame_buffer.get(vehicle_id, [])
        frames = list(buffer) if buffer else []
        frames.append({'frame': fallback_frame, 'box': fallback_box})
        
        best_plate = None
        best_text = None
        best_confidence = 0.0
        best_score = -1.0
        
        for entry in frames:
            plate = self.detect_license_plate(entry['frame'], entry['box'])
            if not plate:
                continue
            
            # Read plate text
            text, ocr_conf = self.ocr.read_plate(plate['image'])
            
            # Score combines detection and OCR confidence
            score = plate['confidence'] * 0.4 + ocr_conf * 0.6
            if text:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_plate = plate['image']
                best_text = text
                best_confidence = ocr_conf
        
        return best_plate, best_text, best_confidence
    
    def _save_evidence(self, output_frame: np.ndarray, frame: np.ndarray,
                      vehicle: Dict, plate_image: Optional[np.ndarray],
                      plate_text: Optional[str]) -> Dict:
        """Save all evidence files for a violation"""
        vid = vehicle['id']
        x1, y1, x2, y2 = vehicle['box']
        
        # 1. Full violation frame
        frame_path = os.path.join(
            OUTPUT_DIRS['evidence'],
            f"violation_frame_{self.total_violations:03d}_frame_{self.frame_count}.jpg"
        )
        cv2.imwrite(frame_path, output_frame)
        
        # 2. Cropped vehicle
        vehicle_crop = frame[y1:y2, x1:x2]
        vehicle_path = os.path.join(
            OUTPUT_DIRS['evidence'],
            f"vehicle_{self.total_violations:03d}_id_{vid}.jpg"
        )
        if vehicle_crop.size > 0:
            cv2.imwrite(vehicle_path, vehicle_crop)
        
        # 3. Citation image
        citation_path = self._generate_citation(
            output_frame, vehicle_crop, plate_image,
            vid, plate_text, vehicle['box']
        )
        
        return {
            'frame_path': frame_path,
            'vehicle_path': vehicle_path,
            'citation_path': citation_path,
        }
    
    def _generate_citation(self, output_frame: np.ndarray, vehicle_crop: np.ndarray,
                          plate_image: Optional[np.ndarray], vehicle_id: int,
                          plate_text: Optional[str], vehicle_box: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Generate a professional citation/evidence image.
        Includes the scene, vehicle crop, plate crop, and violation details.
        """
        try:
            scene = output_frame.copy()
            h, w = scene.shape[:2]
            
            # Create citation canvas
            banner_h = 80
            inset_h = 150
            total_h = h + banner_h + inset_h + 30
            
            citation = np.zeros((total_h, w, 3), dtype=np.uint8)
            citation[:] = (30, 30, 30)  # Dark background
            
            # Place the scene
            citation[banner_h:banner_h + h, 0:w] = scene
            
            # Header with violation details
            x1, y1, x2, y2 = vehicle_box
            video_time = str(timedelta(seconds=int(self.frame_count / self.fps)))
            
            header_lines = [
                f"🚦 RED LIGHT VIOLATION #{self.total_violations:03d}",
                f"Vehicle ID: {vehicle_id} | Plate: {plate_text or 'UNREADABLE'}",
                f"Time: {video_time} | Frame: {self.frame_count}",
            ]
            
            y_pos = 45
            for i, line in enumerate(header_lines):
                color = (0, 0, 255) if i == 0 else (255, 255, 255)
                cv2.putText(citation, line, (15, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_pos += 28
            
            # Inset strip for vehicle and plate
            strip_y = banner_h + h + 15
            x_cursor = 15
            
            def paste_inset(img: np.ndarray, label: str, color: Tuple[int, int, int] = (255, 255, 255)):
                nonlocal x_cursor
                if img is None or img.size == 0:
                    return
                
                ih, iw = img.shape[:2]
                scale = inset_h / ih
                resized = cv2.resize(img, (max(1, int(iw * scale)), inset_h))
                rw = resized.shape[1]
                
                if x_cursor + rw < w - 20:
                    citation[strip_y:strip_y + inset_h, x_cursor:x_cursor + rw] = resized
                    cv2.rectangle(citation, (x_cursor, strip_y),
                                 (x_cursor + rw, strip_y + inset_h),
                                 (100, 100, 100), 2)
                    cv2.putText(citation, label, (x_cursor + 5, strip_y - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    x_cursor += rw + 20
            
            paste_inset(vehicle_crop, "VEHICLE", (0, 255, 0))
            if plate_image is not None:
                paste_inset(plate_image, f"PLATE: {plate_text or 'NOT DETECTED'}", (0, 165, 255))
            
            # Footer
            footer = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(citation, footer, (15, total_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Save citation
            path = os.path.join(
                OUTPUT_DIRS['evidence'],
                f"Citation_{self.total_violations:03d}.jpg"
            )
            cv2.imwrite(path, citation)
            return path
            
        except Exception as e:
            print(f"⚠️ Could not generate citation: {e}")
            return None
    
    def _print_violation(self, record: Dict):
        """Print violation details to console"""
        print("\n" + "=" * 70)
        print(f"🚦 VIOLATION #{record['violation_id']} DETECTED!")
        print("=" * 70)
        print(f"  Vehicle ID    : {record['vehicle_id']}")
        print(f"  Lane          : {record['lane']}")
        print(f"  Frame         : {record['frame']}")
        print(f"  Video Time    : {record['video_time']}")
        print(f"  Plate Number  : {record['plate_text']}")
        print(f"  OCR Confidence: {record['ocr_confidence']:.2f}")
        print("-" * 70)
        print("  Evidence saved:")
        print(f"    Frame    : {os.path.basename(record['frame_path'])}")
        print(f"    Vehicle  : {os.path.basename(record['vehicle_path'])}")
        if record.get('citation_path'):
            print(f"    Citation : {os.path.basename(record['citation_path'])}")
        if record.get('plate_path'):
            print(f"    Plate    : {os.path.basename(record['plate_path'])}")
        print("=" * 70 + "\n")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def draw_annotations(self, frame: np.ndarray, lights: List[Dict],
                         vehicles: List[Dict]) -> np.ndarray:
        """Draw all annotations on the frame"""
        output = frame.copy()
        
        # ── Draw Stop Lines ──
        for lane in self.lanes:
            is_red = lane["red_confirmed"] or lane["state"] == "RED"
            line_color = (0, 0, 255) if is_red else (0, 255, 255)
            
            if lane.get("p1") and lane.get("p2"):
                p1, p2 = lane["p1"], lane["p2"]
                cv2.line(output, p1, p2, line_color, 4)
                label_x, label_y = p1[0], p1[1] - 10
            else:
                y = lane["stop_line_y"]
                x_min, x_max = lane["x_range"]
                x_start = max(0, int(x_min))
                x_end = min(self.frame_width, int(x_max))
                cv2.line(output, (x_start, y), (x_end, y), line_color, 4)
                label_x, label_y = x_start + 10, y - 10
            
            # Lane label
            label = lane['name'].replace(' - ', '\n')
            cv2.putText(output, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
        
        # ── Draw Traffic Lights ──
        for light in lights:
            x1, y1, x2, y2 = light['box']
            cv2.rectangle(output, (x1, y1), (x2, y2), light['color'], 2)
            cv2.putText(output, light['label'].upper(), (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, light['color'], 2)
        
        # ── Draw Vehicles ──
        for vehicle in vehicles:
            vid = vehicle['id']
            x1, y1, x2, y2 = vehicle['box']
            
            is_violator = vid in self.violated_vehicles
            color = (0, 0, 255) if is_violator else (0, 255, 0)
            thickness = 3 if is_violator else 2
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Vehicle ID label
            label = f"ID:{vid}"
            if is_violator:
                label += " 🚨"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # ── Status Panel ──
        self._draw_status_panel(output, vehicles)
        
        # ── Progress Bar ──
        if self.total_frames:
            self._draw_progress_bar(output)
        
        # ── Controls Help ──
        cv2.putText(output, "q=quit  s=save  [/]=move line  n=switch lane",
                   (10, self.frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    def _draw_status_panel(self, output: np.ndarray, vehicles: List[Dict]):
        """Draw the status information panel"""
        panel_h = 34 + 26 * len(self.lanes) + 10
        cv2.rectangle(output, (10, 10), (500, 10 + panel_h), (0, 0, 0), -1)
        cv2.putText(output, "SIGNAL STATUS BY LANE", (20, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        row_y = 58
        for lane in self.lanes:
            is_red = lane["red_confirmed"] or lane["state"] == "RED"
            status = "RED" if is_red else (lane["state"] if lane["state"] != "UNKNOWN" else "NO SIGNAL")
            color = (0, 0, 255) if is_red else (0, 255, 0)
            
            conf_txt = f"{lane['confidence']:.2f}" if lane["confidence"] else "-"
            cv2.putText(output, f"{lane['name']}: {status} (conf {conf_txt})",
                       (20, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            row_y += 26
        
        # Stats
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        video_time = str(timedelta(seconds=int(self.frame_count / self.fps if self.fps else 0)))
        
        panel_top = 10 + panel_h + 10
        cv2.rectangle(output, (10, panel_top), (560, panel_top + 155), (0, 0, 0), -1)
        
        y = panel_top + 23
        cv2.putText(output, f"Frame: {self.frame_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Vehicles: {len(vehicles)}", (200, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Violations: {self.total_violations}", (380, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 255) if self.total_violations > 0 else (255, 255, 255), 1)
        
        y += 27
        cv2.putText(output, f"FPS: {fps:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Time: {video_time}", (200, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(output, f"Total Tracked: {len(self.all_vehicle_ids)}", (380, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        y += 27
        if self.violation_records:
            last = self.violation_records[-1]
            last_txt = f"Last: ID {last['vehicle_id']} Plate: {last['plate_text']}"
        else:
            last_txt = "Last Violation: none yet"
        cv2.putText(output, last_txt, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
    
    def _draw_progress_bar(self, output: np.ndarray):
        """Draw video progress bar"""
        progress = self.frame_count / self.total_frames
        bar_x, bar_y = 10, self.frame_height - 50
        bar_width = 200
        
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        cv2.rectangle(output, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 15),
                     (0, 255, 0), -1)
        cv2.putText(output, f"{progress*100:.1f}%", (bar_x + bar_width + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # ========================================================================
    # DATA PERSISTENCE
    # ========================================================================
    
    def _save_violation_to_csv(self, record: Dict):
        """Append violation record to CSV file"""
        csv_path = os.path.join(OUTPUT_DIRS['base'], "violation_report.csv")
        
        headers = [
            "Violation ID", "Vehicle ID", "Lane", "Plate Number",
            "OCR Confidence", "Frame", "Video Time", "Signal State",
            "Timestamp", "Evidence Frame", "Evidence Vehicle",
            "Evidence Citation", "Plate Image"
        ]
        
        is_new = not os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(headers)
                
                row = [
                    record['violation_id'],
                    record['vehicle_id'],
                    record['lane'],
                    record['plate_text'],
                    f"{record['ocr_confidence']:.2f}",
                    record['frame'],
                    record['video_time'],
                    record['signal_state'],
                    record['timestamp'],
                    os.path.basename(record['frame_path']) if record.get('frame_path') else "",
                    os.path.basename(record['vehicle_path']) if record.get('vehicle_path') else "",
                    os.path.basename(record.get('citation_path', '')),
                    os.path.basename(record.get('plate_path', '')),
                ]
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ Could not save to CSV: {e}")
    
    def save_checkpoint(self):
        """Save checkpoint for long video processing"""
        if not VideoConfig.ENABLE_CHECKPOINTS:
            return
        
        checkpoint = {
            'frame_count': self.frame_count,
            'total_violations': self.total_violations,
            'violation_records': self.violation_records,
            'violated_vehicles': list(self.violated_vehicles),
            'all_vehicle_ids': list(self.all_vehicle_ids),
            'saved_plates': self.saved_plates,
            'lanes': self.lanes,
            'timestamp': datetime.now().isoformat(),
        }
        
        path = os.path.join(
            OUTPUT_DIRS['checkpoints'],
            f"checkpoint_{self.frame_count}.pkl"
        )
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            # Keep only latest 5 checkpoints
            checkpoints = sorted([f for f in os.listdir(OUTPUT_DIRS['checkpoints'])
                                 if f.endswith('.pkl')])
            if len(checkpoints) > 5:
                for old in checkpoints[:-5]:
                    os.remove(os.path.join(OUTPUT_DIRS['checkpoints'], old))
        except Exception as e:
            print(f"⚠️ Could not save checkpoint: {e}")
    
    def save_final_report(self):
        """Save the final violation report"""
        report_path = os.path.join(OUTPUT_DIRS['base'], "violation_report.csv")
        timestamped_path = os.path.join(
            OUTPUT_DIRS['reports'],
            f"violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        headers = [
            "Violation ID", "Vehicle ID", "Lane", "Plate Number",
            "OCR Confidence", "Frame", "Video Time", "Signal State",
            "Timestamp", "Evidence Frame", "Evidence Vehicle",
            "Evidence Citation", "Plate Image"
        ]
        
        for path in [report_path, timestamped_path]:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    for record in self.violation_records:
                        writer.writerow([
                            record['violation_id'],
                            record['vehicle_id'],
                            record['lane'],
                            record['plate_text'],
                            f"{record['ocr_confidence']:.2f}",
                            record['frame'],
                            record['video_time'],
                            record['signal_state'],
                            record['timestamp'],
                            os.path.basename(record['frame_path']) if record.get('frame_path') else "",
                            os.path.basename(record['vehicle_path']) if record.get('vehicle_path') else "",
                            os.path.basename(record.get('citation_path', '')),
                            os.path.basename(record.get('plate_path', '')),
                        ])
            except Exception as e:
                print(f"⚠️ Could not save report: {e}")
        
        print(f"\n📊 Report saved: {report_path}")
        return report_path
    
    def memory_cleanup(self):
        """Clean up stale vehicle data to manage memory"""
        stale_ids = []
        for vid, last_seen in self.vehicle_last_seen.items():
            if self.frame_count - last_seen > VideoConfig.STALE_VEHICLE_FRAMES:
                stale_ids.append(vid)
        
        for vid in stale_ids:
            self.vehicle_frame_buffer.pop(vid, None)
            self.vehicle_history.pop(vid, None)
            self.vehicle_first_seen_side.pop(vid, None)
            self.vehicle_last_seen.pop(vid, None)
        
        if stale_ids:
            print(f"🧹 Pruned {len(stale_ids)} stale vehicle tracks")
    
    # ========================================================================
    # MAIN PROCESSING LOOP
    # ========================================================================
    
    def process_video(self):
        """Main video processing loop"""
        
        # ── Validate video ──
        if not os.path.exists(VIDEO_PATH):
            print(f"❌ Video not found: {VIDEO_PATH}")
            return
        
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"❌ Could not open video: {VIDEO_PATH}")
            return
        
        # ── Get video properties ──
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # ── Setup video writer ──
        output_path = os.path.join(OUTPUT_DIRS['base'], "tracking_output.mp4")
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        # ── Print start info ──
        print("\n" + "=" * 60)
        print("▶️ STARTING VIDEO PROCESSING")
        print("=" * 60)
        print(f"  Video: {os.path.basename(VIDEO_PATH)}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.1f}")
        print(f"  Total Frames: {self.total_frames}")
        print(f"  Duration: {self.total_frames/self.fps/60:.1f} minutes")
        print("=" * 60 + "\n")
        
        # ── Processing loop ──
        self.start_time = time.time()
        read_failures = 0
        last_checkpoint = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                read_failures += 1
                if read_failures <= VideoConfig.FRAME_READ_RETRIES:
                    print(f"⚠️ Frame read failed ({read_failures}/{VideoConfig.FRAME_READ_RETRIES}), retrying...")
                    continue
                print("\n🏁 End of video reached")
                break
            
            read_failures = 0
            self.frame_count += 1
            
            # ── Detection ──
            lights = self.detect_traffic_lights(frame)
            vehicles = self.detect_vehicles(frame)
            
            # ── Update vehicle tracking ──
            self.update_vehicle_history(vehicles, frame)
            
            # ── Draw annotations ──
            output_frame = self.draw_annotations(frame, lights, vehicles)
            
            # ── Check for violations ──
            for vehicle in vehicles:
                self.check_red_light_violation(vehicle, frame, output_frame)
            
            # ── Cleanup ──
            # Remove frame buffer for violated vehicles
            for vid in list(self.vehicle_frame_buffer.keys()):
                if vid in self.violated_vehicles:
                    del self.vehicle_frame_buffer[vid]
            
            # Memory cleanup
            if self.frame_count % VideoConfig.MEMORY_CLEANUP_INTERVAL == 0:
                self.memory_cleanup()
            
            # ── Write output ──
            writer.write(output_frame)
            
            # ── Display ──
            if VideoConfig.SHOW_PREVIEW:
                cv2.imshow("Traffic Violation Detection", output_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️ User stopped processing")
                    break
                elif key == ord('s'):
                    save_path = os.path.join(OUTPUT_DIRS['base'], f"frame_{self.frame_count}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Frame saved: {save_path}")
                elif key == ord(']'):
                    lane = self.lanes[self.active_lane_index]
                    lane["stop_line_y"] = min(self.frame_height - 1, lane["stop_line_y"] + 5)
                    print(f"⬇️ {lane['name']} line: {lane['stop_line_y']}")
                elif key == ord('['):
                    lane = self.lanes[self.active_lane_index]
                    lane["stop_line_y"] = max(0, lane["stop_line_y"] - 5)
                    print(f"⬆️ {lane['name']} line: {lane['stop_line_y']}")
                elif key == ord('n'):
                    self.active_lane_index = (self.active_lane_index + 1) % len(self.lanes)
                    print(f"🔄 Now adjusting: {self.lanes[self.active_lane_index]['name']}")
            
            # ── Progress ──
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed
                progress = self.frame_count / self.total_frames * 100
                eta = (elapsed / self.frame_count * (self.total_frames - self.frame_count)) / 60
                print(f"📊 Frame {self.frame_count}/{self.total_frames} ({progress:.1f}%) "
                      f"| Vehicles: {len(vehicles)} "
                      f"| Violations: {self.total_violations} "
                      f"| FPS: {fps:.1f} "
                      f"| ETA: {eta:.1f}min")
            
            # ── Checkpoint ──
            if (VideoConfig.ENABLE_CHECKPOINTS and 
                self.frame_count - last_checkpoint >= VideoConfig.CHECKPOINT_INTERVAL):
                self.save_checkpoint()
                last_checkpoint = self.frame_count
            
            # ── Test mode ──
            if VideoConfig.TEST_MODE and self.frame_count >= VideoConfig.MAX_TEST_FRAMES:
                print(f"\n🧪 Test mode: processed {self.frame_count} frames")
                break
        
        # ── Cleanup ──
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        # ── Final report ──
        self.save_final_report()
        
        # ── Summary ──
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print("✅ PROCESSING COMPLETE")
        print("=" * 60)
        print(f"  Frames Processed   : {self.frame_count}")
        print(f"  Vehicles Tracked   : {len(self.all_vehicle_ids)}")
        print(f"  Violations Found   : {self.total_violations}")
        print(f"  Plates Saved       : {len(self.saved_plates)}")
        print(f"  Processing Time    : {elapsed/60:.1f} minutes")
        print(f"  Average FPS        : {self.frame_count/elapsed:.1f}")
        print("=" * 60)
        print(f"\n📁 Output saved to: {OUTPUT_DIRS['base']}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        detector = TrafficViolationDetector()
        detector.process_video()
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()