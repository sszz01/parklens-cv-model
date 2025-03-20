# ParkLens-Model

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#-installation)
- [Set up the Environment](#set-up-the-environment)
- [Usage](#-usage)


## Overview
ParkLens-Model is a computer vision model designed to analyze video footage of parking lots to identify empty and occupied parking spots. It is a second repository of the ParkLens-AI App that is used to store folders for the model.
<br>

**Note: This repository goes along with the parking app, the repository of which can be found [here](https://github.com/sszz01/ParkLens-AI)**

**This guide is for the CV model development only. Please refer to other repo for Swift and iOS setup guidelines.**


## Features
- **AI-Powered Parking Detection – Identifies empty and occupied parking spots in real time.** <br>
- **Vehicle Detection – Uses state-of-the-art YOLOv11 model for detecting cars, motorcycles, buses, trucks and more**<br>
- **Live Video Processing – Supports RTMP and RTSP streams for real-time analysis.** <br>
- **Flexible Integration Options –** <br>
    ➤ Manual UI – Easily define parking spots using an interactive polygon-based selection tool. <br>
    ➤ Automated AI Detection (in development) – A custom-trained YOLOv11 model designed to automatically detect empty and occupied spaces.
- **Easy Integration – Designed to work with various camera systems, smart parking solutions, and city infrastructure.**

## What can I use this tool for?
✔️ Smart parking systems
✔️ Shopping malls & commercial lots
✔️ Residential & office buildings
✔️ City-wide parking optimization
✔️ Automated parking enforcement


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
You must have at least OpenCV and Ultralytics installed for the project to be built.<br>


## 3. Set up the environment 
Before running the model, you'll need to configure your environment variables. Here's how to do it:  

### **1. Create Your `.env` File**  
- Copy the example environment file:  

```bash
cp .env_example .env
```

### **2. Edit the `.env` File**  
- Open the `.env` file and add your camera feed URLs. Replace the placeholders with your actual links:  

```env
CAMERA_URL_RTMP=your_camera_rtmp_link
CAMERA_URL_RTSP=your_camera_rtsp_link
RAPI_URL_RTMP=your_raspberry_pi_rtmp_link
```

### **3. Explanation of Variables**  
- **`CAMERA_URL_RTMP`** – The RTMP link of your camera. 
- **`CAMERA_URL_RTSP`** – The RTSP link of your camera. This is an alternative if your camera does not support RTMP, though it may introduce higher latency.  
- **`RAPI_URL_RTMP`** *(Optional)* – If you are using a Raspberry Pi as an external RTMP streaming server, provide its RTMP URL here. This allows the model to push already processed video streams to the Nginx server on Raspberry Pi instead of Docker.


**Note:** If you're unsure which protocol to use, RTMP is generally preferred for real-time applications. However, RTSP remains an option if needed.  

## 🚀 Usage
▶️ **Run the Model**
```bash
python src/main.py
```

As input, select either:
- PC (to use a local video file)
- Camera (to use an RTMP stream)

After video input is specified, select regions of interest by using Tkinter GUI and clicking around the corners of the region to create a tracked polygon.
Then save your preferences by clicking <strong><i>Save</i></strong> button and proceed by choosing a YOLO model from the available options.

**To exit, simply press Q while the video window is open.**

