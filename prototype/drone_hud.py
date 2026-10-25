#Author: Omkar Nalawade
#Date: 03rd October 2026
#Description:   Input and output pipeline for the Gstreamer and OpenCV to work together.
#               AI Injection after reading the frame. Parsing of the YOLOv8s results and passing
#               it to OpenCV for sensible overlay.  
#               And at the end, overlay HUD telemetry information and stream it via RTSP.

#Update:                C001_Rev.B to C002_Rev.B
#Update description:    AI Injection after reading the frame. Parsing of the YOLOv8s results and passing
#                       it to OpenCV for sensible overlay.

#-----------------------------------------------------------------------------------

import cv2
from ultralytics import YOLO

WIDTH = 1024
HEIGHT = 576
FPS = 30
BITRATE = 1500
IS_COLOUR = True
conf_limit = 0.60

# Setting up the YOLOv8s model:

model = YOLO("yolov8s.pt")

# 1. Input: Let libcamera capture natively, then scale and convert to OpenCV's native BGR format
cap_pipeline = (
    f"libcamerasrc ! "
    f"videoconvert ! "
    f"videoscale ! "
    f"video/x-raw, width={WIDTH}, height={HEIGHT}, framerate={FPS}/1, format=BGR ! "
    f"appsink drop=true sync=false"
)
cap = cv2.VideoCapture(cap_pipeline, cv2.CAP_GSTREAMER)

# 2. Output: Added explicit Keyframe limits and disabled B-frames for zero-latency recovery
out_pipeline = (
    f"appsrc ! "
    f"videoconvert ! "
    f"video/x-raw, format=I420 ! "
    f"x264enc tune=zerolatency bitrate={BITRATE} speed-preset=superfast key-int-max={FPS} bframes=0 ! "
    f"rtspclientsink location=rtsp://127.0.0.1:8554/drone protocols=tcp"
)

out = cv2.VideoWriter(out_pipeline, cv2.CAP_GSTREAMER, 0, float(FPS), (WIDTH, HEIGHT), IS_COLOUR)

if not out.isOpened():
    print("FATAL: Output pipeline failed to open!")
    exit()

print("HUD Pipeline Active. Streaming to rtsp://dronepi.local:8554/drone")

while True:
    ret, frame = cap.read() # reading the frame and returning it to "frame". Return is false if it was unable to read. 
    if not ret: 
        print("Failed to grab frame")
        break
    
    # 1. AI INJECTION
    # verbose=False stops the AI from printing text to your terminal every single frame
    results = model(frame, verbose=False)
    
    # Initialize a counter to track how many people are in this specific frame
    person_counter = 1 

    # 2. EXTRACTION
    for r in results:
        boxes = r.boxes  # Extract the bounding box data from the AI's results
        for box in boxes:
            # Extract the Class ID and Confidence Score
            cls_id = int(box.cls[0]) 
            conf = float(box.conf[0])
            
            # YOLO Class 0 is 'person'. We only proceed if it's a person AND confidence is > the conf_limit
            if cls_id == 0 and conf > conf_limit:
                
                # Extract the 4 pixel coordinates and convert them to standard integers
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 3. DYNAMIC OVERLAY
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create the dynamic text (e.g., "Person 1", "Person 2")
                label = f"Person {person_counter}"
                
                # Place the text slightly above the top-left corner of the box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Increment the counter for the next person found in the frame
                person_counter += 1

    # --- HUD OVERLAY LOGIC ---
    height, width, _ = frame.shape
    center_x, center_y = int(width / 2), int(height / 2)

    # Draw Target Crosshairs (Green)
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

    # Draw System Status Text (Red/Yellow)
    cv2.putText(frame, "SYSTEM: ARMED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "TARGETING: STANDBY", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 3. Push the annotated frame to the RTSP server
    out.write(frame)

cap.release()
out.release()