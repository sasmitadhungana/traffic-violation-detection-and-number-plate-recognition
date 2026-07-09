import os
import cv2
from ultralytics import YOLO

# Get the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "traffic_best.pt")
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "testvideo", "test7min.mp4")

# Folder to save the output video
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "traffic_light_output.mp4")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print("Model file not found:")
    print(MODEL_PATH)
    exit()

# Check if video exists
if not os.path.exists(VIDEO_PATH):
    print("Video file not found:")
    print(VIDEO_PATH)
    exit()

print("Loading model...")
model = YOLO(MODEL_PATH)

print("Detected Classes:")
print(model.names)

# Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Failed to open the video.")
    exit()

print("Video opened successfully.")

# Get video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Run traffic light detection
    results = model.predict(frame, conf=0.50, verbose=False)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame
    writer.write(annotated_frame)

    # Display frame
    cv2.imshow("Traffic Light Detection", annotated_frame)

    # Show progress every 100 frames
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

# Release resources
cap.release()
writer.release()
cv2.destroyAllWindows()

print("\nProcessing completed.")
print(f"Total frames processed: {frame_count}")
print(f"Output video saved to:\n{OUTPUT_VIDEO}")