from ultralytics import YOLO

vehicle_model = YOLO("models/vehicle_best.pt")
plate_model = YOLO("models/plate_best.pt")

print("Models loaded successfully")