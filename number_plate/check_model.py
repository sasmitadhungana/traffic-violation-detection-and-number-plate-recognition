from ultralytics import YOLO

model = YOLO(r"F:\best.pt")

print("Classes:")
print(model.names)