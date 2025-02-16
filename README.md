# ParkLens-Model

## Overview
ParkLens-Model is a computer vision model designed to analyze video footage of parking lots to identify empty and occupied parking spots. It is a second repository of the ParkLens-AI App, that is used to store folders for the model. 

## Features
✅ Real-time Object Detection – Detects cars, motorcycles, buses, and trucks.
✅ Vehicle Tracking – YOLOv11 classification
✅ Parking Management – Determines available and occupied parking spots.
✅ Live Stream Support – Works with both recorded videos and real-time camera feeds (soon about to me camera-only)


## ⚙️ Installation  

### 1. Clone the Repository  
```bash
git clone https://github.com/sszz01/parklens-cv-model.git
cd parklens-cv-model
```

## 2. Install the requirements
Make sure you have Python 3.1+ installed, then run:
```bash
pip install -r requirements.txt
```
You must have OpenCV and Ultralytics installed for the project to be build.

# Before running the model, create a .env file and add your camera feed URL:
```bash
CAMERA_URL_RTMP=your_camera_rtmp_link
```
## 🚀 Usage
# ▶️ Run the Model
```bash
python src/parking_management.py
```

You will be asked to select
✔️ PC (to use a local video file)
✔️ Camera (to use an RTMP/RTSP stream)

You will then be asked to choose a YOLO model from the available options.

To exit, simply press Q while the video window is open.

