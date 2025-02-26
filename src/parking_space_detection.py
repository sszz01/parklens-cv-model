from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")
results = model.train(
    data="/Users/ovoievodin/PycharmProjects/parklens-cv-model/data/parking-spaces-dataset/data.yaml",
    epochs=100,
    batch=8,
    imgsz=640,
    device="mps"
)