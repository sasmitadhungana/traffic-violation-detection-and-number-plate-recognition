import os
import cv2
from ultralytics import YOLO

# Project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "plate_best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "testvideo", "test7min.mp4")

print("Loading license plate model...")
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Unable to open the video.")
    exit()

print("Video opened successfully.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # Detect license plates
    results = model.predict(
        frame,
        conf=0.25,      # Lower confidence for testing
        verbose=False
    )

    # Print number of detections
    print("Plates detected:", len(results[0].boxes))

    # Draw detections
    annotated_frame = results[0].plot()

    cv2.imshow("License Plate Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()