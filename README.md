# ParkLens-Model

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#-installation)
- [Set up the Environment](#set-up-the-environment)
- [Usage](#-usage)


## Overview
ParkLens is a computer vision tool designed to analyze video footage of parking lots to identify empty and occupied parking spots. It is a second repository of the ParkLens-AI App that is used to store folders for the model, but it can also be as a standalone tool that can be applied to your solutions.
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
✔️ Smart parking systems <br>
✔️ Shopping malls & commercial lots <br>
✔️ Residential & office buildings <br>
✔️ City-wide parking optimization <br>
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

### **Troubleshooting**
- If you have low recall/precision results on your parking lot footage, try playing around with an IoU (Intersection over Union) threshold in `custom_parking_management.py`. This threshold determines how much overlap between the predicted bounding box and the ground truth is required for a prediction to be considered accurate. By adjusting the IoU threshold, you can tune the model's sensitivity to detect empty or occupied parking spaces more effectively.  
    - **To adjust the IoU threshold:**
      1. Open the `custom_parking_management.py` file.
      2. Locate the line of code where the IoU threshold is set (e.g., `iou_threshold = 0.5`).
      3. Change the value of the threshold to a higher or lower value (e.g., try values between 0.4 to 0.7).
      4. Test the model's performance on your footage and adjust further if needed.
    - **Tip**: A lower threshold may improve recall (finding more true positives), but might also increase false positives. A higher threshold can reduce false positives but may miss some true positives (i.e., lower recall).
  
- **If the model is not detecting parking spots at all:**
    1. Ensure that your camera feed is properly linked in the `.env` file, and that the feed is accessible.
    2. Check that the camera URL (RTMP or RTSP) is valid and that it provides a stable stream.
    3. Confirm that your system meets the necessary hardware and software requirements, particularly for video processing.

- **Performance issues (lag or low frame rate):**
    1. Lower the resolution of the input video to reduce the computational load. This can be done by downscaling the video feed or selecting a lower-resolution stream.
    2. Make sure your environment is set up to use the correct version of OpenCV and YOLOv11, as older versions may cause slower performance.
    3. Try running the model with a smaller input video or a specific camera feed to isolate performance bottlenecks.

- **Error: `ModuleNotFoundError: No module named 'ultralytics'`:**
    1. Make sure that all dependencies in `requirements.txt` are correctly installed by running `pip install -r requirements.txt`.
    2. If the error persists, try manually installing the missing package:
    ```bash
    pip install ultralytics
    ```

- **RTSP feed issues (e.g., lag or no video feed):**
    1. Verify that the RTSP link provided in the `.env` file is correct and accessible.
    2. Some cameras require specific settings or authentication for RTSP streams; check your camera's documentation for any necessary configurations.
    3. Consider switching to an RTMP feed if the RTSP stream is not working reliably.

- **YOLOv11 model not performing as expected:**
    1. Make sure you're using the correct weights file for YOLOv11. You can download the latest version from the official repository or use a custom-trained model if needed.
    2. Test with a different set of test images or video footage to ensure the model is not overfitting or underperforming on specific data types.

- **Issues with Tkinter GUI:**
    1. Ensure that you have the latest version of Tkinter installed. For most systems, you can install it using:
    ```bash
    pip install tk
    ```
    2. If the GUI is not displaying properly, check for any conflicts with your display drivers or try running the model on a different machine.

- **Model crashing during video feed processing:**
    1. Check the logs to identify if there are any specific errors causing the crash. Common issues may relate to memory allocation, feed interruptions, or camera incompatibility.
    2. Ensure your system has sufficient RAM and GPU support for processing video feeds in real time.


