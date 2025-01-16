import cv2
from ultralytics import YOLO

rpi_path = "/home/parkai/Downloads/parking-lot.mp4"
git_path = "../media/parking-lot.mp4"

models = {
    "yolo11n_ncnn": "./yolo11n_ncnn_model",
    "yolo11n": "models/yolo11n.pt",
    "yolo11s": "models/yolo11s.pt",
    "yolo11m": "models/yolo11m.pt",
    "yolo11x": "models/yolo11x.pt",
}

env = input("Choose the device(PC or RPI): ")

print("Available YOLO models:")
for i, (name, path) in enumerate(models.items(), 1):
    print(f"{i}: {name} ({path})")

choice = input("Enter the number of the model you want to use (default: 1): ") or "1"

try:
    selected_model_name = list(models.keys())[int(choice) - 1]
    if selected_model_name.endswith("_ncnn"):
        model = YOLO("yolo11n.pt")
        model.export(format="ncnn")
except (IndexError, ValueError):
    print("Invalid choice. Defaulting to the first model.")
    selected_model_name = list(models.keys())[0]

model_path = models[selected_model_name]
print(f"Selected model: {selected_model_name} ({model_path})")

model = YOLO(model_path)

cap = cv2.VideoCapture(rpi_path if 'rpi' in env.lower() else git_path)

print(f"Processing video from: {rpi_path if 'rpi' in env.lower() else git_path}")

def motion_tracker(frame):
    results = model(frame)

    # lists of detected objects
    cars = []
    motorcycles = []
    buses = []
    trucks = []

    # loop over detected objects
    for result in results[0].boxes:
        x, y, w, h = result.xywh[0]
        cls = result.cls[0]

        if cls == 2:
            cars.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 3:
            motorcycles.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 5:
            buses.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 7:
            trucks.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))

    return cars, motorcycles, buses, trucks

while True:
    success, frame = cap.read()
    if not success:
        break

    new_frame = cv2.resize(frame, (640, 480))
    cars, motorcycles, buses, trucks = motion_tracker(new_frame)

    colors = {
        'car': (0, 0, 255),  # Red
        'motorcycle': (255, 255, 0),  # Blue
        'bus': (255, 255, 255),  # White
        'truck': (255, 165, 0)  # Orange
    }

    # drawing rectangles around detected objects

    for car in cars:
        x, y, w, h = car
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), colors['car'], 2)
        cv2.putText(new_frame, "Car", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['car'], 2)

    for motorcycle in motorcycles:
        x, y, w, h = motorcycle
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), colors['motorcycle'], 2)
        cv2.putText(new_frame, "Motorcycle", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['motorcycle'], 2)

    for bus in buses:
        x, y, w, h = bus
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), colors['bus'], 2)
        cv2.putText(new_frame, "Bus", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['bus'], 2)

    for truck in trucks:
        x, y, w, h = truck
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), colors['truck'], 2)
        cv2.putText(new_frame, "Truck", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['truck'], 2)

    cv2.imshow("Parking Lot", new_frame)
    if cv2.waitKey(32) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
