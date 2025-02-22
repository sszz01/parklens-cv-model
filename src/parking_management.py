import cv2
import os
import time
from ultralytics import solutions, YOLO
from dotenv import load_dotenv
from custom_parking_management import CustomParkingManagement
from coordinates_picker import CustomParkingPtsSelection
from freshest_frame import FreshestFrame
from data.colors import *

#TODO fix polygons being drawn when camera mode is selected
#TODO fix bug where polygons from the old video are drawn on the new video
#TODO add fps option
#TODO put labels around vehicles instead of bboxes

polygon_json_path = "bounding_boxes.json"
git_path = "../media/videos/parking-lot.mp4"
load_dotenv("../env_vars/.env")
camera_url = os.getenv("CAMERA_URL_RTMP") # choose RTMP or RTSP

if not camera_url:
    raise ValueError("CAMERA_URL is not set in .env file or environment")

models = {
    "yolo11n_ncnn": "./yolo11n_ncnn_model",
    "yolo11n": "models/yolo11n.pt",
    "yolo11s": "models/yolo11s.pt",
    "yolo11m": "models/yolo11m.pt",
    "yolo11x": "models/yolo11x.pt",
}

while True:
    env = input("Choose the device (PC or Camera): ").strip().lower()
    if env in ["pc", "camera"]:
        break
    print("Invalid input. Please enter 'pc' or 'camera'.")

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

def get_valid_input(prompt, lower, upper):
    while True:
        try:
            value = int(input(prompt))
            if lower <= value <= upper:
                return value
            else:
                print(f"Invalid input. Please enter a value between {lower} and {upper}")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

width = get_valid_input("Enter the width of the frame (in pixels): ", 640, 1980) or 640 # set 640 by default
height = get_valid_input("Enter the height of the frame (in pixels): ", 480, 1080) or 480 # set 480 by default

stream_res = (width, height)
print(f"Selected resolution: {stream_res}")

model = YOLO(model_path, verbose=False)
model.export(format="coreml")

video_path = git_path if "pc" in env else camera_url

cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
assert cap.isOpened(), f"Error reading {video_path}"

# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

if env == "camera":
    fresh_frame = FreshestFrame(cap)
else:
    fresh_frame = None # the local video is used instead

if not os.path.exists(polygon_json_path):
    print("BBoxes were not found. Please open appeared Tkinter window and save the coordinates to the file.")
    CustomParkingPtsSelection("Select parking ROI", 1980, 1080)

print(f"Processing video from: {video_path}")

def motion_tracker(frame):
    results = model(frame, verbose=False)

    cars = []
    motorcycles = []
    buses = []
    trucks = []

    # loop over detected objects
    for result in results[0].boxes:
        x, y, w, h = result.xywh[0]
        cls = result.cls[0] # get a class of detected object

        # assign those classes to corresponding lists
        if cls == 2:
            cars.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 3:
            motorcycles.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 5:
            buses.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))
        elif cls == 7:
            trucks.append((int(x - w / 2), int(y - h / 2), int(w), int(h)))

    return cars, motorcycles, buses, trucks

parking_manager = CustomParkingManagement(
    model=model_path,  # path to model file
    json_file=polygon_json_path, # path to parking annotations file
    motion_tracker=motion_tracker
)

frame_counter = 0
fps_counter = 0
fps = 0
prev_time = time.time()

while True:
    if fresh_frame is not None:
        frame_counter, frame = fresh_frame.read(seqnumber=frame_counter + 1)
        if fresh_frame is None:
            break
    else:
        success, frame = cap.read()
        if not success:
            break

    new_frame = parking_manager.process_data(frame)
    new_frame = cv2.resize(new_frame, stream_res)

    fps_counter += 1
    if fps_counter % 5 == 0:
        curr_time = time.time()
        fps = 5 / (curr_time - prev_time)
        prev_time = curr_time

    cv2.putText(new_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN, 2)
    cv2.imshow("Parking Lot", new_frame)

    # exit on pressing Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

if fresh_frame is not None:
    fresh_frame.release()
cap.release()
cv2.destroyAllWindows()
