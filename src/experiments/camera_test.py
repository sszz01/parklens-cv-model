import cv2
import os
import subprocess
from ultralytics import YOLO
from data.colors import *
from dotenv import load_dotenv
from src.freshest_frame import FreshestFrame

load_dotenv("../../.env")
camera_url = os.getenv("CAMERA_URL_RTMP") # works on school wifi only

if not camera_url:
    raise ValueError("CAMERA_URL is not set in .env file or environment")

rtmp_url = os.getenv("RAPI_URL_RTMP")

models = {
    "yolo11n": "../models/yolo11n.pt",
    "yolo11s": "../models/yolo11s.pt",
    "yolo11m": "../models/yolo11m.pt",
    "yolo11x": "../models/yolo11x.pt",
}

print("Available YOLO models:")
for i, (name, path) in enumerate(models.items(), 1):
    print(f"{i}: {name} ({path})")

choice = input("Enter the number of the model you want to use (default: 1): ") or "1"

try:
    selected_model_name = list(models.keys())[int(choice) - 1]
    if selected_model_name.endswith("_ncnn"):
        model = YOLO("../archive/yolo11n_ncnn_model/yolo11n.pt")
        model.export(format="ncnn")
except (IndexError, ValueError):
    print("Invalid choice. Defaulting to the first model.")
    selected_model_name = list(models.keys())[0]

model_path = models[selected_model_name]
print(f"Selected model: {selected_model_name} ({model_path})")

model = YOLO(model_path)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fresh_frame = FreshestFrame(cap)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"Processing video from: {camera_url}")

# start the ffmpeg process with optimized params for low latency
ffmpeg_process = subprocess.Popen(
    [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "640x480",
        "-r", "20",
        "-i", "-",
        "-c:v", "libx264",
        "-g", "10",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", "512k",
        "-maxrate", "512k",
        "-bufsize", "256k",
        "-f", "flv",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-avioflags", "direct",
        "-strict", "experimental",
        "-rtbufsize", "64M",
        rtmp_url
    ],
    stdin=subprocess.PIPE
)

# function to track people
def motion_tracker(frame):
    results = model(frame, verbose=False)

    # lists of detected objects
    people, cars = [], []

    # loop over detected objects
    for result in results[0].boxes:
        x, y, w, h = result.xywh[0]
        cls = result.cls[0]

        if cls == 0:
            people.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 2:
            cars.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))

    return people, cars

frame_counter = 0

while True:
    frame_counter, frame = fresh_frame.read(seqnumber=frame_counter + 1)
    if fresh_frame is None:
        break

    new_frame = cv2.resize(frame, (640, 480))
    people, cars = motion_tracker(new_frame)

    # drawing rectangles around detected objects
    for person in people:
        x, y, w, h = person
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), COLOR_RED, 5)
        cv2.putText(new_frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_ORANGE, 2)
    for car in cars:
        x, y, w, h = car
        cv2.rectangle(new_frame, (x, y), (x + w, y + h), COLOR_RED, 2)
        cv2.putText(new_frame, "Car", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEAL, 2)

    cv2.imshow("Frame1", new_frame)
    try:
        ffmpeg_process.stdin.write(new_frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg process broke. Exiting...")
        break

    # press q to quit
    if cv2.waitKey(1) == ord("q"):
        break

if fresh_frame is not None:
    fresh_frame.release()
cap.release()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()
cv2.destroyAllWindows()