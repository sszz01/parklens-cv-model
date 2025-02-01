import cv2
import os
from ultralytics import YOLO
from data.colors import *
from dotenv import load_dotenv

load_dotenv("../env_vars/.env")
camera_url = os.getenv("CAMERA_URL") # works on school wifi only

if not camera_url:
    raise ValueError("CAMERA_URL is not set in .env file or environment")

rpi_path = "/home/parkai/Downloads/parking-lot.mp4"
git_path = "../media/videos/parking-lot.mp4"

models = {
    "yolo11n_ncnn": "./yolo11n_ncnn_model",
    "yolo11n": "models/yolo11n.pt",
    "yolo11s": "models/yolo11s.pt",
    "yolo11m": "models/yolo11m.pt",
    "yolo11x": "models/yolo11x.pt",
}

while True:
    env = input("Choose the device (PC, RPI or Camera): ").strip().lower()
    if env in ["pc", "rpi", "camera"]:
        break
    print("Invalid input. Please enter 'pc', 'rpi' or 'camera.")

print("Available YOLO models:")
for i, (name, path) in enumerate(models.items(), 1):
    print(f"{i}: {name} ({path})")

choice = input("Enter the number of the model you want to use (default: 1): ") or "1"

try:
    selected_model_name = list(models.keys())[int(choice) - 1]
    if selected_model_name.endswith("_ncnn"):
        model = YOLO("yolo11n_ncnn_model/yolo11n.pt")
        model.export(format="ncnn")
except (IndexError, ValueError):
    print("Invalid choice. Defaulting to the first model.")
    selected_model_name = list(models.keys())[0]

model_path = models[selected_model_name]
print(f"Selected model: {selected_model_name} ({model_path})")

model = YOLO(model_path)

video_path = rpi_path if "rpi" in env else git_path if "pc" in env else camera_url

cap = cv2.VideoCapture(video_path)

print(f"Processing video from: {video_path}")

video_writer = cv2.VideoWriter("../media/videos/output/motion_detection.mov", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))

def motion_tracker(frame):
    results = model(frame, verbose=False)

    # lists of detected objects
    people = []

    # loop over detected objects
    for result in results[0].boxes:
        x, y, w, h = result.xywh[0]
        cls = result.cls[0]

        if cls == 0:
            people.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))

    return people

while True:
    success, frame = cap.read()
    if not success:
        break

    new_frame = cv2.resize(frame, (640, 480))
    people = motion_tracker(new_frame)

    # drawing rectangles around detected objects

    for person in people:
        x, y, w, h = person
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), COLOR_RED, 5)
        cv2.putText(new_frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_ORANGE, 2)

    cv2.imshow("Frame", new_frame)
    video_writer.write(new_frame)
    if cv2.waitKey(32) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")