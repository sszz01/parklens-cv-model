import torch
import gc
import os
from ultralytics import YOLO

def mps_clear_memory():
    gc.collect()
    torch.mps.empty_cache()

# some default params
params = dict(
    data="/Users/ovoievodin/PycharmProjects/parklens-cv-model/data/DeteksiParkirKosong.v6i.yolov11/data.yaml",
    epochs=100,
    imgsz=640,
    optimizer="AdamW",
    val=True
)

def set_params():
    print("Which device are you training on?\n1. CPU\n2. GPU\n3. MPS (Apple Silicon Macs only)\n")
    while True:
        choice = input("Select a number: ")
        if choice == "1":
            params["device"] = "cpu"
            params["batch"] = 8
            params["imgsz"] = 640
            params["cache"] = False
            params["workers"] = os.cpu_count() // 2 #use half of the available cores
            break
        elif choice == "2":
            if torch.cuda.is_available():
                # optimizations for CUDA GPU (RTX 3090)
                params["device"] = "cuda"
                params["batch"] = 64
                params["imgsz"] = 1024
                params["cache"] = True
                params["workers"] = os.cpu_count() #use all available cores
                break
            else:
                print("CUDA is not available on your device. Please choose another device.")
        elif choice == "3":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                # optimizations for MPS (Apple Silicon)
                params["device"] = "mps"
                params["batch"] = 16
                params["imgsz"] = 640
                params["cache"] = True
                params["workers"] = 4
                break
            else:
                print("MPS is not available on your device. Please choose another device.")
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
set_params()

if not os.path.exists(params["data"]):
    print(f"Error: Data file not found at {params['data']}")
else:
    if not os.path.exists("models/space_detector/yolo11s.pt"):
        print("Error: Model file not found.")
    else:
        try:
            model = YOLO("models/space_detector/yolo11s.pt")
            results = model.train(**params)
            if params["device"] == "mps":
                mps_clear_memory()
        except Exception as e:
            print(f"An error occurred during training: {e}")