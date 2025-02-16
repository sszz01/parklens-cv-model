# ParkLens-Model

## Overview
ParkLens-Model is a computer vision model designed to analyze video footage of parking lots to identify empty and occupied parking spots. It is a second repository of the ParkLens-AI App that is used to store folders for the model.
<br>

**Note: This repository goes along with the parking app, the repository of which can be found here: https://github.com/sszz01/ParkLens-AI.<br>
This guide is for the CV model development only, please refer to other repo for Swift and iOS setup guidelines.**



## Features
**- Real-time Object Detection – Detects cars, motorcycles, buses, and trucks.**<br>
**- Vehicle Tracking – YOLOv11 classification.**<br>
**- Parking Management – Determines available and occupied parking spots.**<br>
**- Live Stream Support – Works with both recorded videos and real-time camera feeds (soon about to me camera-only).**<br>


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
You must have at least OpenCV and Ultralytics installed for the project to be built.
<br>
**Before running the model, create a .env file and add your camera feed URL**:
```bash
CAMERA_URL_RTMP=your_camera_rtmp_link
```
## 🚀 Usage
▶️ **Run the Model**
```bash
python src/parking_management.py
```

As input, select either:
- PC (to use a local video file)
- Camera (to use an RTMP/RTSP stream)

After video input is specified, select regions of interest by using Tkinter GUI and clicking around the corners of the region to create a tracked polygon.
Then save your preferences by clicking <strong><i>Save</i></strong> button and proceed by choosing a YOLO model from the available options.

*To exit, simply press Q while the video window is open.*

