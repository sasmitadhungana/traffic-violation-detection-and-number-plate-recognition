import asyncio
import os
import sys
import traceback
from typing import Dict, Any


# ============================================================
# PROJECT ROOT
# ============================================================

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT MAIN DETECTOR
# ============================================================

from main_project import TrafficViolationDetector


# ============================================================
# INTEGRATED INFERENCE ENGINE
# ============================================================

class IntegratedInferenceEngine:

    def __init__(self):

        print("=" * 70)
        print("Initializing Integrated Inference Engine")
        print("=" * 70)

        self.detector = None

        self.vehicle_model_path = os.path.join(
            PROJECT_ROOT,
            "models",
            "vehicle_best.pt"
        )

        self.traffic_model_path = os.path.join(
            PROJECT_ROOT,
            "models",
            "traffic_best.pt"
        )

        self.plate_model_path = os.path.join(
            PROJECT_ROOT,
            "models",
            "plate_best.pt"
        )

        print("\nChecking model files...")

        self._check_model(
            "Vehicle Model",
            self.vehicle_model_path
        )

        self._check_model(
            "Traffic Light Model",
            self.traffic_model_path
        )

        self._check_model(
            "License Plate Model",
            self.plate_model_path
        )

        print("\nInference Engine ready.")
        print("=" * 70)


    # ========================================================
    # CHECK MODEL
    # ========================================================

    def _check_model(self, name: str, path: str):

        if os.path.exists(path):

            size_mb = os.path.getsize(path) / (
                1024 * 1024
            )

            print(
                f"[OK] {name}: "
                f"{path} "
                f"({size_mb:.2f} MB)"
            )

        else:

            print(
                f"[WARNING] {name} not found:"
                f"\n          {path}"
            )


    # ========================================================
    # LOAD DETECTOR
    # ========================================================

    def _load_detector(self):

        if self.detector is not None:
            return self.detector

        print("\nLoading TrafficViolationDetector...")

        try:

            self.detector = TrafficViolationDetector()

            print(
                "\nTrafficViolationDetector "
                "loaded successfully."
            )

            return self.detector

        except Exception as e:

            print(
                "\nERROR: Could not load "
                "TrafficViolationDetector"
            )

            print(f"Error: {e}")

            traceback.print_exc()

            raise


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    async def process_video(
        self,
        video_id: str,
        video_path: str,
        db_session=None
    ) -> Dict[str, Any]:

        print("\n" + "=" * 70)
        print("STARTING VIDEO PROCESSING")
        print("=" * 70)

        print(f"Video ID   : {video_id}")
        print(f"Video Path : {video_path}")

        # ----------------------------------------------------
        # CHECK VIDEO
        # ----------------------------------------------------

        if not os.path.exists(video_path):

            error = (
                f"Video file does not exist: "
                f"{video_path}"
            )

            print(error)

            return {
                "status": "Failed",
                "violation_count": 0,
                "violations": [],
                "error": error
            }


        try:

            # ------------------------------------------------
            # LOAD DETECTOR
            # ------------------------------------------------

            detector = self._load_detector()


            # ------------------------------------------------
            # RUN YOUR REAL PIPELINE
            # ------------------------------------------------

            print("\nRunning TrafficViolationDetector...")

            result = detector.process_video(
                video_path
            )


            # ------------------------------------------------
            # HANDLE ASYNC RESULT
            # ------------------------------------------------

            if asyncio.iscoroutine(result):

                result = await result


            # ------------------------------------------------
            # NORMALIZE RESULT
            # ------------------------------------------------

            if result is None:

                result = {}


            if isinstance(result, dict):

                violations = result.get(
                    "violations",
                    []
                )

            elif isinstance(result, list):

                violations = result

            else:

                violations = []


            # ------------------------------------------------
            # FORMAT VIOLATIONS
            # ------------------------------------------------

            normalized_violations = []

            for violation in violations:

                if not isinstance(
                    violation,
                    dict
                ):

                    continue


                item = dict(violation)

                item["video_id"] = video_id


                item.setdefault(
                    "type",
                    "Red Light Violation"
                )

                item.setdefault(
                    "timestamp",
                    "00:00"
                )

                item.setdefault(
                    "confidence",
                    0.0
                )

                item.setdefault(
                    "license_plate",
                    "UNKNOWN"
                )

                item.setdefault(
                    "snapshot_path",
                    ""
                )


                normalized_violations.append(
                    item
                )


            # ------------------------------------------------
            # FINAL RESULT
            # ------------------------------------------------

            print("\n" + "=" * 70)
            print("VIDEO PROCESSING COMPLETED")
            print("=" * 70)

            print(
                f"Total violations: "
                f"{len(normalized_violations)}"
            )


            return {

                "status": "Completed",

                "violation_count":
                    len(normalized_violations),

                "violations":
                    normalized_violations

            }


        except Exception as e:

            print("\n" + "=" * 70)
            print("VIDEO PROCESSING FAILED")
            print("=" * 70)

            print(f"Error: {e}")

            traceback.print_exc()


            return {

                "status": "Failed",

                "violation_count": 0,

                "violations": [],

                "error": str(e)

            }


# ============================================================
# GLOBAL ENGINE
# ============================================================

engine = IntegratedInferenceEngine()