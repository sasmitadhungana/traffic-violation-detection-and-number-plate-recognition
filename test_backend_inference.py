import asyncio
import os
import sys
import traceback

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# IMPORT ENGINE
# ============================================================

from backend.inference_engine import engine


# ============================================================
# TEST CONFIGURATION
# ============================================================

VIDEO_PATH = os.path.join(
    PROJECT_ROOT,
    "backend",
    "uploads",
    "7fadf9b8-cbff-4d14-b1c9-76ee8530fdc1_Traffic_video.mp4"
)

VIDEO_ID = "test-video-001"


# ============================================================
# MAIN TEST
# ============================================================

async def main():

    print("=" * 70)
    print("BACKEND INFERENCE ENGINE TEST")
    print("=" * 70)

    # Check video
    print("\nChecking video file...")

    if not os.path.exists(VIDEO_PATH):
        print("[ERROR] Video file not found:")
        print(VIDEO_PATH)
        return

    print("[OK] Video found:")
    print(VIDEO_PATH)

    # Check engine
    print("\nChecking inference engine...")

    print("Engine type:", type(engine).__name__)

    print("Vehicle model:")
    print(engine.vehicle_model_path)

    print("Traffic model:")
    print(engine.traffic_model_path)

    print("Plate model:")
    print(engine.plate_model_path)

    # Run processing
    print("\nStarting video processing...")
    print("This may take some time.")
    print("=" * 70)

    try:

        results = await engine.process_video(
            video_id=VIDEO_ID,
            video_path=VIDEO_PATH,
            db_session=None
        )

        print("\n")
        print("=" * 70)
        print("PROCESSING FINISHED")
        print("=" * 70)

        print("\nResult:")
        print(results)

        print("\nStatus:")
        print(results.get("status"))

        print("\nViolation count:")
        print(results.get("violation_count"))

        print("\nViolations:")
        for violation in results.get("violations", []):
            print(violation)

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("ERROR DURING INFERENCE")
        print("=" * 70)

        print(str(e))

        traceback.print_exc()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())