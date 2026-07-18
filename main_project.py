"""
Traffic Violation Detection and License Plate Recognition System
================================================================
Professional computer vision system for automated traffic enforcement.

Author: Student Project
Date: July 2026
Version: 2.1 - Fixed video processing hang issues
"""

import os
import sys
import csv
import time
import pickle
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration"""
    traffic_model: str = os.path.join(os.path.dirname(__file__), "models", "traffic_best.pt")
    vehicle_model: str = os.path.join(os.path.dirname(__file__), "models", "vehicle_best.pt")
    plate_model: str = os.path.join(os.path.dirname(__file__), "models", "plate_best.pt")
    
    # Detection thresholds
    traffic_conf: float = 0.10
    vehicle_conf: float = 0.25
    plate_conf: float = 0.15
    
    # Inference settings
    imgsz: int = 640
    device: str = "cpu"
    half: bool = False


@dataclass
class TrackingConfig:
    """Vehicle tracking configuration"""
    track_buffer: int = 90
    match_thresh: float = 0.65
    high_thresh: float = 0.45
    low_thresh: float = 0.1
    new_track_thresh: float = 0.55


@dataclass
class ViolationConfig:
    """Violation detection configuration"""
    red_confirm_frames: int = 3
    state_history_len: int = 5
    state_vote_min: int = 3
    parked_threshold: int = 20
    movement_threshold: float = 3.0
    stop_line_margin: int = 10


@dataclass
class VideoConfig:
    """Video processing configuration"""
    show_preview: bool = True
    save_output: bool = True
    memory_cleanup_interval: int = 200
    stale_vehicle_frames: int = 300
    checkpoint_interval: int = 1000
    enable_checkpoints: bool = True
    max_empty_frames: int = 10  # NEW: Stop after this many empty frames
    frame_timeout_seconds: int = 30  # NEW: Timeout for frame processing


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class SignalState(Enum):
    """Traffic signal states"""
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class Lane:
    """Lane configuration with state"""
    name: str
    x_range: Tuple[int, int]
    stop_line_y: int
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    light_roi: Optional[Tuple[int, int, int, int]] = None
    
    # Runtime state
    state: SignalState = SignalState.UNKNOWN
    state_history: List[str] = field(default_factory=list)
    red_frames: int = 0
    red_confirmed: bool = False
    confidence: float = 0.0
    last_detection: int = 0


@dataclass
class Vehicle:
    """Vehicle tracking data"""
    id: int
    box: Tuple[int, int, int, int]
    center_x: int
    center_y: int
    bottom_y: int
    confidence: float
    first_seen_side: Optional[str] = None
    position_history: List[Tuple[int, int]] = field(default_factory=list)
    frame_buffer: List[Dict] = field(default_factory=list)
    violated: bool = False
    plate_text: Optional[str] = None
    plate_confidence: float = 0.0


@dataclass
class TrafficLight:
    """Traffic light detection"""
    box: Tuple[int, int, int, int]
    label: str
    confidence: float
    center_x: float
    center_y: float
    color: Tuple[int, int, int]


@dataclass
class ViolationRecord:
    """Violation record for reporting"""
    violation_id: int
    vehicle_id: int
    lane: str
    frame: int
    video_time: str
    plate_text: str
    ocr_confidence: float
    signal_state: str
    timestamp: str
    frame_path: str
    vehicle_path: str
    citation_path: str
    plate_path: Optional[str]


# ============================================================================
# TRAFFIC LIGHT DETECTOR
# ============================================================================

class TrafficLightDetector:
    """Traffic light detection and tracking"""
    
    def __init__(self, model: YOLO, config: ModelConfig):
        self.model = model
        self.config = config
        self.cached_detections: List[TrafficLight] = []
        self.frame_count: int = 0
        self.total_red: int = 0
        self.total_green: int = 0
        self.total_yellow: int = 0
    
    def detect(self, frame: np.ndarray) -> List[TrafficLight]:
        """Detect traffic lights in frame"""
        self.frame_count += 1
        
        try:
            results = self.model.predict(
                frame,
                conf=self.config.traffic_conf,
                imgsz=self.config.imgsz,
                device=self.config.device,
                half=self.config.half,
                verbose=False
            )
        except Exception as e:
            print(f"Traffic light detection error: {e}")
            return []
        
        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[class_id].lower()
                
                # Classify signal color
                if "red" in label:
                    label_type = "red"
                    color = (0, 0, 255)
                    self.total_red += 1
                elif "green" in label:
                    label_type = "green"
                    color = (0, 255, 0)
                    self.total_green += 1
                elif "yellow" in label or "amber" in label:
                    label_type = "yellow"
                    color = (0, 255, 255)
                    self.total_yellow += 1
                else:
                    continue
                
                detections.append(TrafficLight(
                    box=(x1, y1, x2, y2),
                    label=label_type,
                    confidence=confidence,
                    center_x=(x1 + x2) / 2,
                    center_y=(y1 + y2) / 2,
                    color=color
                ))
        
        # Cache detections for stability
        if detections:
            self.cached_detections = detections
        
        return detections
    
    def get_stats(self) -> Dict[str, int]:
        """Get detection statistics"""
        return {
            'red': self.total_red,
            'green': self.total_green,
            'yellow': self.total_yellow,
            'total': self.total_red + self.total_green + self.total_yellow
        }


# ============================================================================
# VEHICLE DETECTOR
# ============================================================================

class VehicleDetector:
    """Vehicle detection and tracking"""
    
    def __init__(self, model: YOLO, config: ModelConfig, tracker_config: TrackingConfig):
        self.model = model
        self.config = config
        self.tracker_config = tracker_config
        self.tracker_cfg_path = self._create_tracker_config()
        
        # Vehicle tracking
        self.vehicles: Dict[int, Vehicle] = {}
        self.frame_count: int = 0
        self.total_tracked: int = 0
    
    def _create_tracker_config(self) -> str:
        """Create ByteTrack configuration file"""
        config_path = os.path.join(os.path.dirname(__file__), "bytetrack_custom.yaml")
        
        config_content = f"""
tracker_type: bytetrack
track_high_thresh: {self.tracker_config.high_thresh}
track_low_thresh: {self.tracker_config.low_thresh}
new_track_thresh: {self.tracker_config.new_track_thresh}
track_buffer: {self.tracker_config.track_buffer}
match_thresh: {self.tracker_config.match_thresh}
fuse_score: True
"""
        try:
            with open(config_path, "w") as f:
                f.write(config_content)
            return config_path
        except Exception:
            return "bytetrack.yaml"
    
    def detect(self, frame: np.ndarray) -> List[Vehicle]:
        """Detect and track vehicles"""
        self.frame_count += 1
        
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.config.vehicle_conf,
                imgsz=self.config.imgsz,
                device=self.config.device,
                half=self.config.half,
                tracker=self.tracker_cfg_path,
                verbose=False
            )
        except Exception:
            try:
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=self.config.vehicle_conf,
                    imgsz=self.config.imgsz,
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
                
                # Filter small detections
                if (y2 - y1) < 30 or (x2 - x1) < 30:
                    continue
                
                vehicle = Vehicle(
                    id=vehicle_id,
                    box=(x1, y1, x2, y2),
                    center_x=(x1 + x2) // 2,
                    center_y=(y1 + y2) // 2,
                    bottom_y=y2,
                    confidence=confidence
                )
                
                # Update tracking
                if vehicle_id in self.vehicles:
                    old = self.vehicles[vehicle_id]
                    vehicle.position_history = old.position_history[-30:]
                    vehicle.frame_buffer = old.frame_buffer[-8:]
                    vehicle.first_seen_side = old.first_seen_side
                    vehicle.violated = old.violated
                    vehicle.plate_text = old.plate_text
                    vehicle.plate_confidence = old.plate_confidence
                
                vehicles.append(vehicle)
                
                # Update or create vehicle tracking
                if vehicle_id not in self.vehicles:
                    self.total_tracked += 1
                self.vehicles[vehicle_id] = vehicle
        
        return vehicles
    
    def update_vehicle(self, vehicle: Vehicle, frame: np.ndarray, frame_count: int):
        """Update vehicle tracking data"""
        # Update position history
        vehicle.position_history.append((vehicle.center_x, vehicle.bottom_y))
        if len(vehicle.position_history) > 30:
            vehicle.position_history.pop(0)
        
        # Update frame buffer
        vehicle.frame_buffer.append({'frame': frame.copy(), 'box': vehicle.box})
        if len(vehicle.frame_buffer) > 8:
            vehicle.frame_buffer.pop(0)
        
        # Update last seen
        self.vehicles[vehicle.id] = vehicle
    
    def cleanup_stale(self, frame_count: int, threshold: int = 300):
        """Remove stale vehicles"""
        stale_ids = []
        for vid, vehicle in self.vehicles.items():
            if not vehicle.position_history:
                stale_ids.append(vid)
            elif frame_count - len(vehicle.position_history) > threshold:
                stale_ids.append(vid)
        
        for vid in stale_ids:
            del self.vehicles[vid]
        
        return len(stale_ids)


# ============================================================================
# LICENSE PLATE READER
# ============================================================================

class LicensePlateReader:
    """License plate detection and OCR"""
    
    def __init__(self, model: YOLO, config: ModelConfig):
        self.model = model
        self.config = config
        self.frame_count: int = 0
        
        # Initialize OCR
        self._ocr = None
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False
            )
            print("✓ OCR initialized successfully")
        except Exception as e:
            print(f"! OCR initialization failed: {e}")
    
    def detect_plate(self, frame: np.ndarray, vehicle_box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Detect license plate in vehicle region"""
        if self.model is None:
            return None
            
        x1, y1, x2, y2 = vehicle_box
        
        # Extract vehicle region with padding
        padding = 60
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(frame.shape[1], x2 + padding)
        y2p = min(frame.shape[0], y2 + padding)
        
        vehicle_region = frame[y1p:y2p, x1p:x2p]
        if vehicle_region.size == 0:
            return None
        
        # Try multiple scales
        scales = [1.0, 1.5, 2.0]
        best_plate = None
        best_confidence = 0.0
        
        for scale in scales:
            if scale != 1.0:
                h, w = vehicle_region.shape[:2]
                if h * scale < 50 or w * scale < 50:
                    continue
                scaled = cv2.resize(vehicle_region, (int(w * scale), int(h * scale)), cv2.INTER_CUBIC)
            else:
                scaled = vehicle_region
            
            try:
                results = self.model.predict(
                    scaled,
                    conf=self.config.plate_conf,
                    device=self.config.device,
                    half=self.config.half,
                    verbose=False
                )
            except Exception:
                continue
            
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = float(box.conf[0])
                    if confidence > best_confidence:
                        px1, py1, px2, py2 = map(int, box.xyxy[0])
                        
                        if scale != 1.0:
                            px1 = int(px1 / scale)
                            py1 = int(py1 / scale)
                            px2 = int(px2 / scale)
                            py2 = int(py2 / scale)
                        
                        margin = 10
                        abs_x1 = max(0, x1p + px1 - margin)
                        abs_y1 = max(0, y1p + py1 - margin)
                        abs_x2 = min(frame.shape[1], x1p + px2 + margin)
                        abs_y2 = min(frame.shape[0], y1p + py2 + margin)
                        
                        plate_img = frame[abs_y1:abs_y2, abs_x1:abs_x2]
                        if plate_img.size > 0:
                            best_plate = plate_img
                            best_confidence = confidence
        
        return best_plate
    
    def read_plate(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Read text from plate using OCR"""
        if plate_image is None or plate_image.size == 0:
            return None, 0.0
        
        if self._ocr is None:
            return None, 0.0
        
        try:
            # Preprocess plate image
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Convert to RGB
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            
            # Run OCR
            results = self._ocr.ocr(rgb, cls=True)
            
            if not results or not results[0]:
                return None, 0.0
            
            # Extract text
            texts = []
            confidences = []
            
            for line in results[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Clean text
                    cleaned = ''.join(c for c in text if c.isalnum())
                    cleaned = cleaned.strip().upper()
                    
                    if cleaned and len(cleaned) >= 2:
                        texts.append(cleaned)
                        confidences.append(confidence)
            
            if texts:
                combined = ''.join(texts)
                avg_conf = sum(confidences) / len(confidences)
                
                # Validate
                if len(combined) <= 12 and len(combined) >= 2:
                    return combined, avg_conf
            
            return None, 0.0
            
        except Exception as e:
            return None, 0.0


# ============================================================================
# VIOLATION DETECTOR
# ============================================================================

class ViolationDetector:
    """Red light violation detection"""
    
    def __init__(self, config: ViolationConfig, lanes: List[Lane]):
        self.config = config
        self.lanes = lanes
        self.violated_vehicles: set = set()
        self.violation_records: List[ViolationRecord] = []
        self.total_violations: int = 0
        self.crossing_events: int = 0
        self.frame_count: int = 0
    
    def update_lane_state(self, lane: Lane, detections: List[TrafficLight]):
        """Update lane state based on traffic light detections"""
        # Find detections in lane's ROI
        if lane.light_roi:
            rx1, ry1, rx2, ry2 = lane.light_roi
            pool = [
                d for d in detections
                if rx1 <= d.center_x <= rx2 and ry1 <= d.center_y <= ry2
            ]
        else:
            pool = detections
        
        # Get best detection
        if pool:
            best = max(pool, key=lambda d: d.confidence)
            lane.state = SignalState(best.label.upper())
            lane.confidence = best.confidence
            lane.last_detection = self.frame_count
            
            if lane.state == SignalState.RED:
                lane.red_frames += 1
            else:
                lane.red_frames = 0
        else:
            # Keep previous state if we had recent detections
            if self.frame_count - lane.last_detection > 30:
                lane.state = SignalState.UNKNOWN
                lane.confidence = 0.0
                lane.red_frames = 0
        
        # Update history
        lane.state_history.append(lane.state.value)
        if len(lane.state_history) > self.config.state_history_len:
            lane.state_history.pop(0)
        
        # Confirm red state
        if lane.state == SignalState.RED:
            # Need consecutive red frames
            lane.red_confirmed = lane.red_frames >= self.config.red_confirm_frames
    
    def check_violation(self, vehicle: Vehicle, lanes: List[Lane]) -> Optional[ViolationRecord]:
        """Check if vehicle committed a violation"""
        # Already violated
        if vehicle.violated or vehicle.id in self.violated_vehicles:
            return None
        
        # Must have been seen approaching
        if vehicle.first_seen_side != "BEFORE":
            return None
        
        # Check if parked
        if self._is_parked(vehicle):
            return None
        
        # Check if crossed stop line
        if not self._crossed_stop_line(vehicle, lanes):
            return None
        
        self.crossing_events += 1
        
        # Get lane and check red light
        lane = self._get_lane(vehicle, lanes)
        if not lane or not lane.red_confirmed:
            return None
        
        # Violation detected
        vehicle.violated = True
        self.violated_vehicles.add(vehicle.id)
        self.total_violations += 1
        
        return self._create_record(vehicle, lane)
    
    def _is_parked(self, vehicle: Vehicle) -> bool:
        """Check if vehicle is parked/stationary"""
        history = vehicle.position_history
        if len(history) < self.config.parked_threshold:
            return False
        
        window = history[-self.config.parked_threshold:]
        xs = [p[0] for p in window]
        ys = [p[1] for p in window]
        displacement = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return displacement < 5.0
    
    def _crossed_stop_line(self, vehicle: Vehicle, lanes: List[Lane]) -> bool:
        """Check if vehicle crossed stop line"""
        history = vehicle.position_history
        if len(history) < 3:
            return False
        
        # Check movement direction (moving upward)
        prev_y = history[-2][1]
        curr_y = history[-1][1]
        if prev_y - curr_y < self.config.movement_threshold:
            return False
        
        # Get lane and check crossing
        lane = self._get_lane(vehicle, lanes)
        if not lane:
            return False
        
        prev_dist = self._distance_to_line(lane, history[-2][0], history[-2][1])
        curr_dist = self._distance_to_line(lane, history[-1][0], history[-1][1])
        
        # Crossed from below to above
        return prev_dist < 0 <= curr_dist
    
    def _get_lane(self, vehicle: Vehicle, lanes: List[Lane]) -> Optional[Lane]:
        """Get lane for vehicle position"""
        for lane in lanes:
            x_min, x_max = lane.x_range
            if x_min <= vehicle.center_x <= x_max:
                return lane
        return None
    
    def _distance_to_line(self, lane: Lane, x: int, y: int) -> float:
        """Calculate signed distance to stop line"""
        x1, y1 = lane.p1
        x2, y2 = lane.p2
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length == 0:
            return lane.stop_line_y - y
        return ((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)) / length
    
    def _create_record(self, vehicle: Vehicle, lane: Lane) -> ViolationRecord:
        """Create violation record"""
        return ViolationRecord(
            violation_id=self.total_violations,
            vehicle_id=vehicle.id,
            lane=lane.name,
            frame=self.frame_count,
            video_time=str(timedelta(seconds=self.frame_count // 30)),
            plate_text=vehicle.plate_text or "UNREADABLE",
            ocr_confidence=vehicle.plate_confidence,
            signal_state=lane.state.value,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_path="",
            vehicle_path="",
            citation_path="",
            plate_path=""
        )


# ============================================================================
# MAIN SYSTEM CLASS
# ============================================================================

class TrafficViolationSystem:
    """Main traffic violation detection system"""
    
    def __init__(self, config_dir: str = None):
        """Initialize the system"""
        self.config_dir = config_dir or os.path.dirname(__file__)
        self.setup_directories()
        self.load_config()
        self.load_models()
        self.setup_lanes()
        self.setup_components()
        
        print("\n" + "=" * 70)
        print("TRAFFIC VIOLATION DETECTION SYSTEM v2.1")
        print("=" * 70)
        print(f"  Output Directory: {self.output_dir}")
        print(f"  Vehicle Model: {os.path.basename(self.model_config.vehicle_model)}")
        print(f"  Traffic Model: {os.path.basename(self.model_config.traffic_model)}")
        print("=" * 70 + "\n")
    
    def setup_directories(self):
        """Create output directories"""
        self.output_dir = os.path.join(self.config_dir, "output")
        self.evidence_dir = os.path.join(self.output_dir, "violations")
        self.plates_dir = os.path.join(self.output_dir, "violations", "plates")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        
        for d in [self.output_dir, self.evidence_dir, self.plates_dir, 
                  self.reports_dir, self.checkpoints_dir]:
            os.makedirs(d, exist_ok=True)
    
    def load_config(self):
        """Load configuration"""
        self.model_config = ModelConfig()
        self.tracking_config = TrackingConfig()
        self.violation_config = ViolationConfig()
        self.video_config = VideoConfig()
    
    def load_models(self):
        """Load YOLO models"""
        print("Loading models...")
        self.models = {}
        
        try:
            self.models['vehicle'] = YOLO(self.model_config.vehicle_model)
            print(f"  ✓ Vehicle model loaded")
        except Exception as e:
            print(f"  ✗ Failed to load vehicle model: {e}")
            sys.exit(1)
        
        try:
            self.models['traffic'] = YOLO(self.model_config.traffic_model)
            print(f"  ✓ Traffic model loaded")
        except Exception as e:
            print(f"  ✗ Failed to load traffic model: {e}")
            self.models['traffic'] = None
        
        try:
            self.models['plate'] = YOLO(self.model_config.plate_model)
            print(f"  ✓ Plate model loaded")
        except Exception as e:
            print(f"  ✗ Failed to load plate model: {e}")
            self.models['plate'] = None
        
        # Store references for backward compatibility
        self.vehicle_model = self.models.get('vehicle')
        self.traffic_model = self.models.get('traffic')
        self.plate_model = self.models.get('plate')
    
    def setup_lanes(self):
        """Setup lanes for the video"""
        # For 1280x720 video - adjust these for your video
        self.lanes = [
            Lane(
                name="Lane 1 - Left",
                x_range=(0, 320),
                stop_line_y=560,
                p1=(0, 560),
                p2=(320, 560),
                light_roi=(50, 50, 200, 150)
            ),
            Lane(
                name="Lane 2 - Center Left",
                x_range=(320, 640),
                stop_line_y=560,
                p1=(320, 560),
                p2=(640, 560),
                light_roi=(250, 50, 400, 150)
            ),
            Lane(
                name="Lane 3 - Center Right",
                x_range=(640, 960),
                stop_line_y=560,
                p1=(640, 560),
                p2=(960, 560),
                light_roi=(450, 50, 600, 150)
            ),
            Lane(
                name="Lane 4 - Right",
                x_range=(960, 1280),
                stop_line_y=560,
                p1=(960, 560),
                p2=(1280, 560),
                light_roi=(650, 50, 800, 150)
            ),
        ]
    
    def setup_components(self):
        """Setup detection components"""
        self.traffic_detector = TrafficLightDetector(
            self.models.get('traffic'), self.model_config
        )
        
        self.vehicle_detector = VehicleDetector(
            self.models.get('vehicle'), self.model_config, self.tracking_config
        )
        
        self.plate_reader = LicensePlateReader(
            self.models.get('plate'), self.model_config
        ) if self.models.get('plate') else None
        
        self.violation_detector = ViolationDetector(
            self.violation_config, self.lanes
        )
    
    def process_video(self, video_path: str = None):
        """Process video for violations - FIXED VERSION with proper error handling"""
        
        # Default video path if none provided
        if not video_path:
            video_path = os.path.join(self.config_dir, "videos", "testvideo", "test7min.mp4")
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            print(f"   Please check the path and try again.")
            return
        
        # Open video with validation
        print(f"\n📹 Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            print(f"   The file might be corrupted or in an unsupported format.")
            return
        
        # Validate video has frames
        first_frame_ret, first_frame = cap.read()
        if not first_frame_ret or first_frame is None:
            print(f"❌ Video appears to be empty or corrupt: {video_path}")
            cap.release()
            return
        
        # Reset video capture
        cap.release()
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {width}x{height}, {fps:.1f} fps, {total_frames} frames")
        
        # Setup video writer
        output_path = os.path.join(self.output_dir, "tracking_output.mp4")
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )
        
        # Processing variables
        frame_count = 0
        start_time = time.time()
        consecutive_empty_frames = 0
        max_empty_frames = self.video_config.max_empty_frames
        max_frames = total_frames + 50  # Allow small buffer beyond total frames
        
        print("\n🔍 Processing video...")
        print("-" * 60)
        print(f"   Total frames expected: {total_frames}")
        print(f"   Max empty frames allowed: {max_empty_frames}")
        print("-" * 60)
        
        # Main processing loop
        while True:
            # Read frame with timeout check
            ret, frame = cap.read()
            
            # CRITICAL FIX: Check if frame reading failed
            if not ret or frame is None:
                consecutive_empty_frames += 1
                print(f"⚠️  Warning: Empty frame #{frame_count + 1}, consecutive: {consecutive_empty_frames}/{max_empty_frames}")
                
                # Stop if too many empty frames
                if consecutive_empty_frames >= max_empty_frames:
                    print(f"🛑 Too many empty frames ({consecutive_empty_frames}), stopping processing...")
                    break
                
                # Skip to next iteration
                continue
            
            # Reset empty frame counter on successful read
            consecutive_empty_frames = 0
            frame_count += 1
            
            # Safety check - prevent infinite loop
            if frame_count > max_frames:
                print(f"⚠️  Reached maximum frame limit ({max_frames}), stopping...")
                break
            
            # Update frame counter in detectors
            self.violation_detector.frame_count = frame_count
            
            # Detect traffic lights
            traffic_lights = self.traffic_detector.detect(frame)
            
            # Update lane states
            for lane in self.lanes:
                self.violation_detector.update_lane_state(lane, traffic_lights)
            
            # Detect vehicles
            vehicles = self.vehicle_detector.detect(frame)
            
            # Update vehicle tracking
            for vehicle in vehicles:
                self.vehicle_detector.update_vehicle(vehicle, frame, frame_count)
                
                # First seen side
                if vehicle.first_seen_side is None:
                    lane = self.violation_detector._get_lane(vehicle, self.lanes)
                    if lane:
                        dist = self.violation_detector._distance_to_line(
                            lane, vehicle.center_x, vehicle.bottom_y
                        )
                        vehicle.first_seen_side = "BEFORE" if dist < 0 else "AFTER"
                
                # Check for plate
                if self.plate_reader and not vehicle.plate_text:
                    plate_img = self.plate_reader.detect_plate(frame, vehicle.box)
                    if plate_img is not None:
                        plate_text, conf = self.plate_reader.read_plate(plate_img)
                        if plate_text:
                            vehicle.plate_text = plate_text
                            vehicle.plate_confidence = conf
            
            # Check for violations
            for vehicle in vehicles:
                record = self.violation_detector.check_violation(
                    vehicle, self.lanes
                )
                if record:
                    # Save evidence
                    record = self.save_evidence(record, frame, vehicle)
                    print(f"\n🚨 *** VIOLATION #{record.violation_id} DETECTED ***")
                    print(f"   Vehicle: ID {record.vehicle_id}")
                    print(f"   Plate: {record.plate_text}")
                    print(f"   Lane: {record.lane}")
                    print(f"   Frame: {record.frame}")
            
            # Cleanup stale vehicles periodically
            if frame_count % self.video_config.memory_cleanup_interval == 0:
                self.vehicle_detector.cleanup_stale(
                    frame_count, self.video_config.stale_vehicle_frames
                )
            
            # Draw annotations
            annotated_frame = self.draw_annotations(frame, traffic_lights, vehicles)
            
            # Save output
            if self.video_config.save_output:
                writer.write(annotated_frame)
            
            # Show preview
            if self.video_config.show_preview:
                cv2.imshow("Traffic Violation Detection", annotated_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  User stopped processing")
                    break
                elif key == ord('s'):
                    save_path = os.path.join(self.output_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"💾 Frame saved: {save_path}")
            
            # Progress update
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                stats = self.traffic_detector.get_stats()
                
                # Calculate estimated remaining time
                if fps_actual > 0:
                    remaining_frames = total_frames - frame_count
                    remaining_seconds = remaining_frames / fps_actual
                    remaining_minutes = remaining_seconds / 60
                    eta_str = f"{remaining_minutes:.1f} min"
                else:
                    eta_str = "calculating..."
                
                print(f"📊 Frame {frame_count}/{total_frames} ({progress:.1f}%) "
                      f"| Vehicles: {len(vehicles):3d} "
                      f"| Lights: {stats['total']:3d} (R:{stats['red']:3d}, G:{stats['green']:3d}) "
                      f"| Violations: {self.violation_detector.total_violations:3d} "
                      f"| FPS: {fps_actual:.1f} "
                      f"| ETA: {eta_str}")
        
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
        # Final report
        self.save_report()
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ PROCESSING COMPLETE")
        print("=" * 70)
        print(f"  Frames Processed: {frame_count}")
        print(f"  Vehicles Tracked: {self.vehicle_detector.total_tracked}")
        print(f"  Violations Found: {self.violation_detector.total_violations}")
        print(f"  Processing Time: {elapsed/60:.1f} minutes")
        if elapsed > 0:
            print(f"  Average FPS: {frame_count/elapsed:.1f}")
        print(f"  Output Saved: {output_path}")
        print(f"  Report Saved: {self.output_dir}/violation_report.csv")
        print("=" * 70)
    
    def draw_annotations(self, frame: np.ndarray, traffic_lights: List[TrafficLight], 
                         vehicles: List[Vehicle]) -> np.ndarray:
        """Draw annotations on frame"""
        output = frame.copy()
        
        # Draw traffic light ROIs
        for lane in self.lanes:
            if lane.light_roi:
                x1, y1, x2, y2 = lane.light_roi
                color = (0, 255, 255) if lane.red_confirmed else (100, 100, 100)
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
        
        # Draw traffic light detections
        for light in traffic_lights:
            x1, y1, x2, y2 = light.box
            color = light.color
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{light.label.upper()} {light.confidence:.2f}"
            cv2.putText(output, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw lanes and stop lines
        for lane in self.lanes:
            is_red = lane.red_confirmed
            line_color = (0, 0, 255) if is_red else (0, 255, 255)
            cv2.line(output, lane.p1, lane.p2, line_color, 3)
            
            # Lane label
            status = "RED" if is_red else (lane.state.value if lane.state != SignalState.UNKNOWN else "---")
            label = f"{lane.name}\n{status}"
            cv2.putText(output, label, (lane.p1[0], lane.p1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2)
        
        # Draw vehicles
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.box
            color = (0, 0, 255) if vehicle.violated else (0, 255, 0)
            thickness = 3 if vehicle.violated else 2
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            label = f"ID:{vehicle.id}"
            if vehicle.violated:
                label += " 🚨"
            if vehicle.plate_text:
                label += f" {vehicle.plate_text}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Status panel
        self.draw_status_panel(output, vehicles, traffic_lights)
        
        return output
    
    def draw_status_panel(self, output: np.ndarray, vehicles: List[Vehicle], 
                          traffic_lights: List[TrafficLight]):
        """Draw status panel"""
        height = output.shape[0]
        
        # Panel dimensions
        panel_x, panel_y = 10, 10
        panel_w, panel_h = 500, 40 + 26 * len(self.lanes) + 10
        
        # Background
        cv2.rectangle(output, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        
        # Title
        cv2.putText(output, "🚦 SIGNAL STATUS", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Lane statuses
        y_pos = panel_y + 50
        for lane in self.lanes:
            is_red = lane.red_confirmed
            status = "🔴 RED" if is_red else (lane.state.value if lane.state != SignalState.UNKNOWN else "---")
            color = (0, 0, 255) if is_red else (0, 255, 0)
            conf = f"{lane.confidence:.2f}" if lane.confidence > 0 else "-"
            cv2.putText(output, f"{lane.name}: {status} (conf={conf})",
                       (panel_x + 10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_pos += 26
        
        # Stats panel
        stats_y = panel_y + panel_h + 10
        stats = self.traffic_detector.get_stats()
        violations = self.violation_detector.total_violations
        crossings = self.violation_detector.crossing_events
        
        cv2.rectangle(output, (panel_x, stats_y), 
                     (panel_x + 600, stats_y + 130), (0, 0, 0), -1)
        
        y_pos = stats_y + 25
        cv2.putText(output, f"Frame: {self.violation_detector.frame_count}", 
                   (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, f"Vehicles: {len(vehicles)}", 
                   (panel_x + 200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, f"🚨 Violations: {violations}", 
                   (panel_x + 400, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 255) if violations > 0 else (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(output, f"🔴 Red: {stats['red']}", 
                   (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(output, f"🟢 Green: {stats['green']}", 
                   (panel_x + 200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(output, f"Crossings: {crossings}", 
                   (panel_x + 400, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(output, f"Tracked: {self.vehicle_detector.total_tracked}", 
                   (panel_x + 10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, f"Total Detections: {stats['total']}", 
                   (panel_x + 200, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(output, "q=quit  s=save frame", 
                   (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def save_evidence(self, record: ViolationRecord, frame: np.ndarray, 
                      vehicle: Vehicle) -> ViolationRecord:
        """Save evidence for violation"""
        x1, y1, x2, y2 = vehicle.box
        
        # Frame with annotations
        frame_path = os.path.join(
            self.evidence_dir,
            f"violation_{record.violation_id:03d}_frame.jpg"
        )
        cv2.imwrite(frame_path, frame)
        
        # Vehicle crop
        vehicle_crop = frame[y1:y2, x1:x2]
        vehicle_path = os.path.join(
            self.evidence_dir,
            f"violation_{record.violation_id:03d}_vehicle.jpg"
        )
        if vehicle_crop.size > 0:
            cv2.imwrite(vehicle_path, vehicle_crop)
        
        # Plate crop
        plate_path = None
        if self.plate_reader:
            plate_img = self.plate_reader.detect_plate(frame, vehicle.box)
            if plate_img is not None:
                plate_path = os.path.join(
                    self.plates_dir,
                    f"violation_{record.violation_id:03d}_plate.jpg"
                )
                cv2.imwrite(plate_path, plate_img)
        
        # Citation
        citation_path = self.generate_citation(frame, vehicle_crop, plate_img, record)
        
        # Update record
        record.frame_path = frame_path
        record.vehicle_path = vehicle_path
        record.plate_path = plate_path
        record.citation_path = citation_path
        
        return record
    
    def generate_citation(self, frame: np.ndarray, vehicle_crop: np.ndarray,
                         plate_img: np.ndarray, record: ViolationRecord) -> str:
        """Generate citation image"""
        h, w = frame.shape[:2]
        
        # Create citation
        banner_h = 80
        inset_h = 150
        total_h = h + banner_h + inset_h + 30
        
        citation = np.zeros((total_h, w, 3), dtype=np.uint8)
        citation[:] = (30, 30, 30)
        
        # Place frame
        citation[banner_h:banner_h + h, 0:w] = frame
        
        # Header
        header = [
            f"🚨 RED LIGHT VIOLATION #{record.violation_id:03d}",
            f"Vehicle: ID {record.vehicle_id} | Plate: {record.plate_text}",
            f"Time: {record.video_time} | Frame: {record.frame}"
        ]
        
        y_pos = 45
        for i, text in enumerate(header):
            color = (0, 0, 255) if i == 0 else (255, 255, 255)
            cv2.putText(citation, text, (15, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 28
        
        # Insets
        strip_y = banner_h + h + 15
        x_pos = 15
        
        def paste_inset(img: np.ndarray, label: str):
            nonlocal x_pos
            if img is None or img.size == 0:
                return
            
            ih, iw = img.shape[:2]
            scale = inset_h / ih
            resized = cv2.resize(img, (max(1, int(iw * scale)), inset_h))
            rw = resized.shape[1]
            
            if x_pos + rw < w - 20:
                citation[strip_y:strip_y + inset_h, x_pos:x_pos + rw] = resized
                cv2.rectangle(citation, (x_pos, strip_y),
                             (x_pos + rw, strip_y + inset_h),
                             (100, 100, 100), 2)
                cv2.putText(citation, label, (x_pos + 5, strip_y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                x_pos += rw + 20
        
        paste_inset(vehicle_crop, "VEHICLE")
        if plate_img is not None:
            paste_inset(plate_img, f"PLATE: {record.plate_text}")
        
        # Footer
        footer = f"Generated: {record.timestamp}"
        cv2.putText(citation, footer, (15, total_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Save
        path = os.path.join(
            self.evidence_dir,
            f"citation_{record.violation_id:03d}.jpg"
        )
        cv2.imwrite(path, citation)
        
        return path
    
    def save_report(self):
        """Save final violation report"""
        report_path = os.path.join(self.output_dir, "violation_report.csv")
        
        # CSV headers
        headers = [
            "Violation ID", "Vehicle ID", "Lane", "Plate Number",
            "OCR Confidence", "Frame", "Video Time", "Signal State",
            "Timestamp", "Frame Path", "Vehicle Path", "Citation Path"
        ]
        
        try:
            with open(report_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for record in self.violation_detector.violation_records:
                    writer.writerow([
                        record.violation_id,
                        record.vehicle_id,
                        record.lane,
                        record.plate_text,
                        f"{record.ocr_confidence:.2f}",
                        record.frame,
                        record.video_time,
                        record.signal_state,
                        record.timestamp,
                        os.path.basename(record.frame_path) if record.frame_path else "",
                        os.path.basename(record.vehicle_path) if record.vehicle_path else "",
                        os.path.basename(record.citation_path) if record.citation_path else "",
                    ])
            
            print(f"\n📄 Report saved: {report_path}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER
# ============================================================================

class TrafficViolationDetector:
    """
    Main traffic violation detection class.
    This wrapper maintains compatibility with existing code.
    """
    
    def __init__(self, model_paths=None):
        """Initialize the detector with optional custom model paths"""
        if model_paths:
            # Update model paths if provided
            if 'traffic' in model_paths:
                ModelConfig.traffic_model = model_paths['traffic']
            if 'vehicle' in model_paths:
                ModelConfig.vehicle_model = model_paths['vehicle']
            if 'plate' in model_paths:
                ModelConfig.plate_model = model_paths['plate']
        
        # Initialize the actual system
        self.system = TrafficViolationSystem()
        
        # Copy references for backward compatibility
        self.models = self.system.models if hasattr(self.system, 'models') else {}
        self.lanes = self.system.lanes
        self.frame_count = 0
        self.total_violations = 0
        self.violation_records = []
        
        print("\n" + "=" * 70)
        print("🚦 TRAFFIC VIOLATION DETECTOR INITIALIZED")
        print("=" * 70)
    
    def process_video(self, video_path=None):
        """Process video for violations"""
        self.system.process_video(video_path)
        
        # Update references
        self.frame_count = self.system.violation_detector.frame_count
        self.total_violations = self.system.violation_detector.total_violations
        self.violation_records = self.system.violation_detector.violation_records
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for real-time detection.
        Returns detection results.
        """
        # This is for real-time processing
        # Detect traffic lights
        traffic_lights = self.system.traffic_detector.detect(frame)
        
        # Update lane states
        for lane in self.system.lanes:
            self.system.violation_detector.update_lane_state(lane, traffic_lights)
        
        # Detect vehicles
        vehicles = self.system.vehicle_detector.detect(frame)
        
        # Update vehicle tracking
        for vehicle in vehicles:
            self.system.vehicle_detector.update_vehicle(
                vehicle, frame, self.system.violation_detector.frame_count
            )
            
            # First seen side
            if vehicle.first_seen_side is None:
                lane = self.system.violation_detector._get_lane(vehicle, self.system.lanes)
                if lane:
                    dist = self.system.violation_detector._distance_to_line(
                        lane, vehicle.center_x, vehicle.bottom_y
                    )
                    vehicle.first_seen_side = "BEFORE" if dist < 0 else "AFTER"
            
            # Check for plate
            if self.system.plate_reader and not vehicle.plate_text:
                plate_img = self.system.plate_reader.detect_plate(frame, vehicle.box)
                if plate_img is not None:
                    plate_text, conf = self.system.plate_reader.read_plate(plate_img)
                    if plate_text:
                        vehicle.plate_text = plate_text
                        vehicle.plate_confidence = conf
        
        # Check for violations
        violations = []
        for vehicle in vehicles:
            record = self.system.violation_detector.check_violation(
                vehicle, self.system.lanes
            )
            if record:
                violations.append(record)
        
        # Update frame count
        self.system.violation_detector.frame_count += 1
        self.frame_count = self.system.violation_detector.frame_count
        self.total_violations = self.system.violation_detector.total_violations
        self.violation_records = self.system.violation_detector.violation_records
        
        return {
            'frame': self.frame_count,
            'vehicles': len(vehicles),
            'traffic_lights': len(traffic_lights),
            'violations': violations,
            'lane_states': [{'name': l.name, 'state': l.state.value, 'red_confirmed': l.red_confirmed} 
                           for l in self.system.lanes],
            'plate_texts': [{'id': v.id, 'plate': v.plate_text} for v in vehicles if v.plate_text]
        }
    
    def draw_annotations(self, frame: np.ndarray) -> np.ndarray:
        """Draw annotations on frame"""
        return self.system.draw_annotations(frame, [], [])


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 70)
        print("🚦 TRAFFIC VIOLATION DETECTION SYSTEM")
        print("=" * 70)
        
        # Create detector
        detector = TrafficViolationDetector()
        
        # Specify video path
        video_path = "videos/Tests.mp4"
        
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"\n⚠️  Video not found: {video_path}")
            print("   Please check the path and try again.")
            print("   Make sure the video file exists in the 'videos' folder.")
        else:
            print(f"\n▶️  Starting processing...")
            detector.process_video(video_path)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("👋 Press Enter to exit...")
    input()