# YOLO-Project-Person-and-Vehicle-Recognition
A project for identifying and locating vehicles and people that works with AI and CV
# ADEEB AI VISION
## Multi-Device Computer Vision for Real-Time Detection, Tracking, and Counting of People and Vehicles

ADEEB AI VISION is a Python-based computer vision project that uses YOLO technology to detect and track people and specific vehicle categories through live camera streams. The project provides a responsive Arabic-language web interface that adapts to smartphones and laptops and can display up to **four connected devices in total** on the server dashboard.

## 1. Project Overview

ADEEB AI VISION is an analytical dashboard for live camera streams, designed to make basic object detection and traffic monitoring accessible and easy to use directly through a web browser. When a user opens the web page, they are prompted to enter a display name and select whether the current device is a **smartphone** or a **laptop**. The user interface then automatically adapts to the selected device type.

The system processes camera frames on a Python server, draws color-coded bounding boxes around supported object classes, assigns tracking IDs, calculates current counts for each class, and maintains session-wide detection totals. It also records events when tracked objects cross a predefined counting line.

## 2. Responsive Interface and Device-Specific Design

- Displays a welcome screen prompting users to enter a display name and select their device type.
- Provides a compact, mobile-optimized interface for smartphones.
- Provides laptops with a monitoring and control dashboard featuring a device grid that accommodates up to four devices.
- Supports up to **four devices connected to the server simultaneously**. For example, one laptop dashboard and three smartphones can occupy all four available slots.
- Displays each connected device's name, type, camera status, live video stream, and detected object counts.
- Supports video streaming from both laptop cameras and smartphone cameras.
- Allows users on supported smartphones to switch between the front-facing and rear-facing cameras.
- Transmits camera frames to the Python server using Socket.IO.
- Runs YOLO inference on the server and returns annotated frames to the smartphone and connected laptop dashboards.

## 3. Detection, Tracking, and Counting

- Detects people, cars, motorcycles, buses, and trucks.
- Draws a different-colored bounding box for each supported object class.
- Adds an English class label, tracking ID, and confidence score to each detected object.
- Displays the current number of detected objects, categorized by class.
- Tracks newly detected objects and updates session-wide detection totals.
- Allows users to draw a counting line from either the laptop dashboard or the smartphone interface by selecting two points on the video.
- Records line-crossing events with a timestamp, object class, tracking ID, device name, and direction (entry or exit).
- Provides an option to reset tracking totals and line-crossing counts.

## 4. Interface and Networking

- Features a dark-mode user interface with green accents and a right-to-left (RTL) Arabic layout.
- Displays a live connection-status indicator and the capacity of connected devices.
- Uses Socket.IO event-based communication between browser clients and the server.
- Integrates the server, processing logic, event handlers, and embedded HTML/CSS/JavaScript interface into a single Python file.

## 5. Detection Classes and Colors

The default model uses the COCO class IDs listed below. Bounding-box colors in the processed video are defined using OpenCV's **BGR** channel order in Python. The dashboard uses separate highlight colors defined in CSS. The motorcycle color currently differs between the video bounding box and the statistics card.

| Object Class | Display Name | Bounding Box / Dashboard Color |
|---|---|---|
| Person | Person / شخص | Magenta (light purple) in the video; purple in the dashboard (`#DF55FF`). |
| Car | Car / سيارة | Green in the video and dashboard (`#00E650` in the dashboard). |
| Motorcycle | Motorcycle / دراجة نارية | Blue/cyan in the video (`#0082FF` based on the current BGR value); orange in the dashboard (`#FF9D35`). |
| Bus | Bus / باص | Yellow in the video and dashboard (`#FFD642` in the dashboard). |
| Truck | Truck / شاحنة | Red in the video and dashboard (`#FF4545` in the dashboard). |

**Note:** The motorcycle bounding-box color and dashboard highlight color are not currently identical. These color mappings are configured in the `CLASS_INFO` variable and the frontend's `classMeta` configuration. Both can be updated if consistent colors across the video and dashboard are desired.
