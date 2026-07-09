from ultralytics import YOLO

model = YOLO(r"F:\best.pt")

results = model.predict(
    source="test.jpg",
    conf=0.5,
    save=True
)

print("License plate detection completed")