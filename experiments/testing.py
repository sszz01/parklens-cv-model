from ultralytics import YOLO

model = YOLO('models/yolo11s.pt')
print(model.names)
